import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterator, List, Union
from datetime import datetime
from keras import Model
from keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint


def run_model(
    model_name: str,
    model_function: Model,
    lr: float, n_epochs, n_workers,
    train_generator: Iterator,
    validation_generator: Iterator,
    test_generator: Iterator,
) -> History:

    callbacks = get_callbacks(model_name)
    model = model_function

    history = model.fit(
        train_generator,
        epochs=n_epochs, 
        validation_data=validation_generator,
        callbacks=callbacks,
        workers=n_workers # adjust this according to the number of CPU cores of your machine
    )

    model.evaluate(
        test_generator,
        callbacks=callbacks,
    )
    return history  # type: ignore


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
        "logs/scalars/" + model_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )  # create a folder for each model.
    tensorboard_callback = TensorBoard(log_dir=logdir)
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy",
        patience=10,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=2,
        mode="max",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    model_checkpoint_callback = ModelCheckpoint(
        "./data/models/" + model_name,
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,  # save the best model
        mode="max",
        save_freq="epoch",  # save every epoch
    )  # saving eff_net takes quite a bit of time
    return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback]