import tensorflow.keras
import sys
from tensorflow.keras import layers, Model, Sequential, Input
from tensorflow.keras.activations import elu
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
# from keras_radam import RAdam

from efficientnet.tfkeras import EfficientNetB4, EfficientNetB5
from data.dataset import loadTeacherDatabase, loadTESTDatabase, loadDatabaseUnlabeled, loadStudentDatabase, \
    loadNewDatabase
from data.const import IMG_SIZE, NUM_CLASSES_TRAIN, LEARNING_RATE, UNFREEZE_LEARNING_RATE, \
    N_EPOCHS, N_WORKERS, TOP_DROPOUT_RATE, MODEL_NAME, HIST_PATH, PLOT_PATH, WEIGHTS, PATIENCE, SYS_PATH, \
    N_EPOCHS_UNFREEZE
from model_func import run_model, save_plot_history, plot_acc, test_model, teacher_predict_unlabeled, \
    save_labeled_results
from group_norm import GroupNormalization
from radam import RAdam

sys.path.append(SYS_PATH)

def get_optimizer(learning_rate):
    # optimizer = tensorflow.keras.optimizers.Adam(lr=learning_rate) #old
    optimizer = tensorflow.keras.optimizers.SGD(lr=learning_rate)
    # optimizer = tensorflow.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    # optimizer = tensorflow.keras.optimizers.RMSprop(learning_rate=learning_rate)
    # optimizer = tensorflow.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07)
    return optimizer

def build_model(model_name, learning_rate, top_dropout_rate, num_classes, weights) -> Model:

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # x = img_augmentation(inputs)
    model = EfficientNetB5(include_top=False, input_tensor=inputs, weights=weights)

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # print('1!!!! AAAAAAAA')
    # model.summary()
    plot_model(model, to_file=PLOT_PATH + ".jpg", show_shapes=True)

    # Compile
    model = tensorflow.keras.Model(inputs, outputs, name=model_name) #"EfficientNet"

    optimizer = get_optimizer(learning_rate)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # print('2!!!! AAAAAAAA')
    # model.summary()

    return model


def unfreeze_model(model, learning_rate):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = get_optimizer(learning_rate)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    print('3!!!! AAAAAAAA')
    # model.summary()
    return model

def run():
    # train_generator, validation_generator, test_generator = loadTeacherDatabase(False)
    train_generator, validation_generator, test_generator = loadNewDatabase() # run_new_dataset

    # with strategy.scope():
    # model = load_model('models/24_new_db_effnet_b7_epoch-08_val_loss-8.98_val_acc-0.03.hdf5')
    model = build_model(MODEL_NAME, LEARNING_RATE, TOP_DROPOUT_RATE, NUM_CLASSES_TRAIN, WEIGHTS)
    model.load_weights(tf.train.latest_checkpoint(os.path.dirname('models/24_new_db_effnet_b7_epoch-08_val_loss-8.98_val_acc-0.03.hdf5')))
    model.summary()

    eff_net_history = run_model(
        model_name=MODEL_NAME,
        hist_path=HIST_PATH,
        model_function=model,
        n_epochs=N_EPOCHS, n_workers=N_WORKERS,
        patience=PATIENCE,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator
    )

    model = unfreeze_model(model, UNFREEZE_LEARNING_RATE)

    eff_net_history = run_model(
        model_name=MODEL_NAME+'after_unfreeze',
        hist_path=HIST_PATH+'after_unfreeze',
        model_function=model,
        n_epochs=N_EPOCHS_UNFREEZE, n_workers=N_WORKERS,
        patience=PATIENCE,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator
    )

    
def test():
    test_generator = loadTESTDatabase(False)
    # TODO change it!!!!!!!
    checkpoint = '11_effnet_b4/11_effnet_b4_epoch-50_val_loss-1.46_val_acc-0.56.hdf5'
    # checkpoint = 'gold/5_eff_net_b4_imagenet_weights_epoch-01_val_loss-16.12_val_acc-0.00.hdf5' #TODO name showld be from const
    test_model(checkpoint,
               # build_model(MODEL_NAME, LEARNING_RATE, TOP_DROPOUT_RATE, NUM_CLASSES_TEST),
               test_generator=test_generator)


def teacher_labeling():
    unlabeled_generator = loadDatabaseUnlabeled()
    # TODO change it!!!!!!!
    checkpoint = '12_effnet_b4/12_effnet_b4_model.hdf5'
    teacher_predict_unlabeled(checkpoint,
               # build_model(MODEL_NAME, LEARNING_RATE, TOP_DROPOUT_RATE, NUM_CLASSES_TEST),
               unlabeled_generator=unlabeled_generator)


def save_label_results():
    unlabeled_generator = loadDatabaseUnlabeled()
    save_labeled_results(unlabeled_generator)

def run_student_resume():
    train_generator, validation_generator, test_generator = loadStudentDatabase()

    # with strategy.scope():
    # model = build_model(MODEL_NAME, LEARNING_RATE, TOP_DROPOUT_RATE, NUM_CLASSES_TRAIN, WEIGHTS)
    # model = unfreeze_model(model, UNFREEZE_LEARNING_RATE)

    checkpoint = 'student_12_effnet_b4/student_12_effnet_b4_model.hdf5'

    eff_net_history = resume_training(
        checkpoint = checkpoint,
        model_name=MODEL_NAME,
        hist_path=HIST_PATH,
        # model_function=model,
        n_epochs=N_EPOCHS, n_workers=N_WORKERS,
        patience=PATIENCE,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator
    )
    
    


if __name__ == '__main__':
    print('AA')
    run()
    # b5()
    # teacher_labeling()
    # save_label_results()
    # run_student()
    # run_student_resume()


