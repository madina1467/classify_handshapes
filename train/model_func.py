import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator, List, Union
from datetime import datetime
from keras import Model
from keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
import sys
from sklearn.metrics import classification_report

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
    # test_loss, test_acc = model.evaluate_generator(
    #     test_generator, len(test_generator) // BATCH_SIZE,
    # )
    # result = loaded_model.predict(test_image/255)

    # print('test_loss: ', test_loss, ', test_acc: ', test_acc)

    predictions = model.predict_generator(test_generator,steps = len(test_generator) // BATCH_SIZE )

    # preds_test = np.zeros((len(test_generator), test_generator.shape[1]), dtype=np.float)
    # print('preds_test', predictions.shape, preds_test_fold.shape, predictions)

    # print('sum:', sum(preds_test_fold[1]))
    # preds = predictions.round(decimals=2)
    # for p in preds:
    #     print(p)
    test_labels = test_generator.classes
    y_pred = np.argmax(predictions, axis=-1)
    print('test_labels: ', len(test_labels), 'y_pred: ', y_pred.shape)
    # print(classification_report(test_labels, y_pred))
    print('AAAAAA')

    print('test_generator[1]: ', len(test_generator), 'predictions: ', len(predictions))
    # print(classification_report(test_generator[2], predictions))

    # print(classification_report(test_generator.argmax(axis=1), predictions.argmax(axis=1), target_names=list(range(0, 60))))

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
        verbose=0,
        save_best_only=False,  # TODO CHECK TRUE later, save the best model
        mode='auto',
        save_weights_only=False,
        period=SAVE_PERIOD  # save every SAVE_PERIOD epoch
    )  # saving eff_net takes quite a bit of time
    return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback]
