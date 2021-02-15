import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator, List, Union
from datetime import datetime
from keras import Model
from keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
import sys
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append('/home/kenny/PycharmProjects/classify_handshapes')
from data.const import BATCH_SIZE, SAVE_PERIOD


def run_model(
        model_name: str,
        model_function: Model,
        n_epochs, n_workers,
        train_generator: Iterator,
        validation_generator: Iterator,
        test_generator: Iterator,
) -> History:
    callbacks = get_callbacks(model_name)
    model = model_function

    history = model.fit_generator(
        train_generator,
        epochs=n_epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        steps_per_epoch=len(train_generator) // BATCH_SIZE,
        validation_steps=len(validation_generator) // BATCH_SIZE,
        workers=n_workers  # TODO adjust this according to the number of CPU cores of your machine
    )

    model.evaluate_generator(
        test_generator, len(test_generator) // BATCH_SIZE,
    )
    return history  # type: ignore


def test_model(file_name, test_generator: Iterator):
    #  model: Model,
    model = load_model('models/'+file_name)
    # model.load_weights('models/'+file_name)

    test_loss, test_acc = model.evaluate_generator(
        test_generator, len(test_generator) // BATCH_SIZE,
    )
    print('test_loss: ', test_loss, ', test_acc: ', test_acc)

    # ValueError: Error when checking target: expected pred to have shape (51,) but got array with shape (45,)
    # result = loaded_model.predict(test_image/255)


    #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    # pred = model.predict_generator(test_generator, steps=len(test_generator) // BATCH_SIZE, verbose=1)
    #
    # preds_cls_idx = pred.argmax(axis=-1)
    # idx_to_cls = {v: k for k, v in test_generator.class_indices.items()}
    # preds_cls = np.vectorize(idx_to_cls.get)(preds_cls_idx)
    # true_cls = test_generator.classes
    # filenames_to_cls = list(zip(test_generator.filenames, preds_cls, true_cls))
    # df = pd.DataFrame(filenames_to_cls)
    # df.to_csv(r'1_test_result.csv')
    # print(sum(preds_cls_idx[:, 0] == true_cls) / len(true_cls))
    #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

    # results = pd.DataFrame({"Filename": test_generator.filenames,
    #                         "Predictions": preds_cls,
    #                         "TRUE class": true_cls})
    # results.to_csv(r'1_test_result.csv')
    #

    # # 5
    # true_labels = test_generator.classes
    # preds = pred.argmax(axis=-1)
    # print("!!! FFFFFFF")
    # print(sum(preds[:, 0] == true_labels) / len(true_labels))

    # 2
    # predicted_class_indices=np.argmax(pred,axis=1)
    # labels = (test_generator.class_indices)
    # labels = dict((v, k) for k, v in labels.items())
    # predictions = [labels[k] for k in predicted_class_indices]
    #
    # filenames = test_generator.filenames
    # results = pd.DataFrame({"Filename": filenames,
    #                         "Predictions": predictions})
    # print(results)

    # 3
    # test_data = []
    # test_labels = []
    # batch_index = 0
    #
    # while batch_index <= test_generator.batch_index:
    #     data = next(test_generator)
    #     test_data.append(data[0])
    #     test_labels.append(data[1])
    #     batch_index = batch_index + 1
    #
    # test_data_array = np.asarray(test_data)
    # test_labels_array = np.asarray(test_labels)
    #
    # data_count, batch_count, w, h, c = test_data_array.shape
    #
    # test_data_array = np.reshape(test_data_array, (data_count * batch_count, w, h, c))
    # test_labels_array = np.reshape(test_labels_array, (data_count * batch_count, -1))
    #
    # cm = confusion_matrix(test_labels_array.argmax(axis=1), probabilities.argmax(axis=1))
    #
    # print(cm)



def plot_results(model_history_eff_net: History):
    plt.plot(model_history_eff_net.history["accuracy"])
    plt.plot(model_history_eff_net.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig("training_validation.png")
    plt.show()


def get_callbacks(model_name: str) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
    logdir = (
            'logs/scalars/' + model_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )  # create a folder for each model.
    tensorboard_callback = TensorBoard(log_dir=logdir)
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=2,
        mode='auto',
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )
    # weights.{epoch:02d}-{val_loss:.2f}.hdf5   #_{epoch:02d}.ckpt
    model_checkpoint_callback = ModelCheckpoint(
        'models/' + model_name + '_weights_epoch-{epoch:02d}_val_loss-{val_loss:.2f}.hdf5',
        monitor='val_loss',
        verbose=1,
        save_best_only=False,  # TODO CHECK TRUE later, save the best model
        mode='auto',
        save_weights_only=False,
        period=SAVE_PERIOD  # save every SAVE_PERIOD epoch
    )  # saving eff_net takes quite a bit of time
    return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback]
