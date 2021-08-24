import math
import matplotlib.pyplot as plt
import sys
import dill as pickle
import numpy as np
import pandas as pd
import atexit
import signal
from functools import partial

from typing import Iterator, List, Union
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from data.const import BATCH_SIZE, SAVE_PERIOD, MODEL_PATH, LOG_PATH, PLOT_PATH, CLASSES, HIST_PLOT_PATH, SYS_PATH, \
    MODEL_NAME, ITERATION, LABELS_PATH, MODEL_CSV_HIST_PATH
import os
from os import path

# from train.metric import Metrics

sys.path.append(SYS_PATH)

def run_model(
        model_name: str,
        hist_path:str,
        model_function: Model,
        n_epochs, n_workers, patience,
        train_generator: Iterator,
        validation_generator: Iterator
) -> History:
    callbacks = get_callbacks(model_name, patience)
    model = model_function

    history = model.fit(
        train_generator,
        epochs=n_epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        workers=n_workers  # TODO adjust this according to the number of CPU cores of your machine
    )

    # model.evaluate_generator(
    #     test_generator, len(test_generator) // BATCH_SIZE,
    # )

    save_history(hist_path, history)
    save_plot_history(hist_path)
    plot_acc(hist_path)

    return history  # type: ignore


def done_function(fileName, toSave):
    with open(fileName, 'wb') as f:
        np.save(f, toSave)
    with open(fileName, 'rb') as f:
        a = np.load(f)
    print(a)
    print('WWWWWWWWWWWWW')

def teacher_predict_unlabeled(file_name, unlabeled_generator: Iterator):
    model = load_model('models/'+file_name)
    unlabeled_generator.reset()

    # pred222 = np.array([])
    # atexit.register(done_function, fileName='testtesttesttest22.npy', toSave=pred222)

    # for sig in signal.valid_signals():
    #     print(f'{sig.value}: signal.{sig.name},')
    #     signal.signal(sig, partial(done_function, fileName='testtesttesttest22.npy', toSave=pred222))

    # signal.signal(signal.SIGTERM, partial(done_function, fileName='testtesttesttest22.npy', toSave=pred222))
    # signal.signal(signal.SIGINT, partial(done_function, fileName='testtesttesttest22.npy', toSave=pred222))

    pred = model.predict_generator(unlabeled_generator, steps=len(unlabeled_generator), verbose=1)

    with open('results/labeling/' + ITERATION + '_' + MODEL_NAME + '_teacher_unlabeled_result.npy', 'wb') as f:
        np.save(f, pred)

    predicted_class_indices = np.argmax(pred, axis=1)

    results = pd.DataFrame({"Filename": unlabeled_generator.filenames,
                            "Predictions": predicted_class_indices,
                            "TRUE class": unlabeled_generator.classes})

    results.to_csv('results/labeling/' + ITERATION + '_' + MODEL_NAME +'_teacher_unlabeled_result.csv')

    label_map = (unlabeled_generator.class_indices)
    label_map = dict((v,k) for k,v in label_map.items())
    predictions = [label_map[k] for k in predicted_class_indices]
    results['TEST'] = predictions

    results.to_csv('results/labeling/' + ITERATION + '_' + MODEL_NAME + '_teacher_unlabeled_result.csv')

def save_labeled_results(unlabeled_generator):
    with open(os.path.join(LABELS_PATH, ITERATION + '_' + MODEL_NAME + '_teacher_unlabeled_result.npy'), 'rb') as f:
        pred = np.load(f)

    # predicted_class_indices = np.argmax(pred, axis=1)
    # results = pd.DataFrame({"Filename": unlabeled_generator.filenames,
    #                         "Predictions": predicted_class_indices})

    i = 0
    results = pd.DataFrame(columns=['Filename', 'Prediction', 'PredPercent', 'Prediction2', 'PredPercent2', 'Prediction3', 'PredPercent3'])
    res05 = pd.DataFrame(columns=['Filename', 'Prediction', 'PredPercent', 'Prediction2', 'PredPercent2', 'Prediction3', 'PredPercent3'])
    for row in pred:

        predictions_idx = np.argpartition(row, len(row) - 3)[-3:]

        results = results.append({'Filename': unlabeled_generator.filenames[i],
                                  'Prediction': predictions_idx[2], 'PredPercent': row[predictions_idx[2]],
                                  'Prediction2': predictions_idx[1], 'PredPercent2': row[predictions_idx[1]],
                                  'Prediction3': predictions_idx[0], 'PredPercent3': row[predictions_idx[0]]
                                  }, ignore_index=True)
        if row[predictions_idx[2]] > 0.5:
            res05 = res05.append({'Filename': unlabeled_generator.filenames[i],
                                  'Prediction': predictions_idx[2], 'PredPercent': row[predictions_idx[2]],
                                  'Prediction2': predictions_idx[1], 'PredPercent2': row[predictions_idx[1]],
                                  'Prediction3': predictions_idx[0], 'PredPercent3': row[predictions_idx[0]]
                                  }, ignore_index=True)

        i = i + 1
    results.to_csv(os.path.join(LABELS_PATH, ITERATION + '_' + MODEL_NAME + '_teacher_unlabeled_result.csv'))
    res05.to_csv(os.path.join(LABELS_PATH, ITERATION + '_' + MODEL_NAME + '_teacher_unlabeled_result_0.5.csv'))


def test_model(file_name, test_generator: Iterator):
    model = load_model('models/'+file_name)
    # evaluation = model.evaluate_generator(
    #     test_generator, len(test_generator) // BATCH_SIZE)
    # plot_test_results(model, evaluation)

    # result = loaded_model.predict(test_image/255)


    #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    # test_generator.reset()
    # pred = model.predict_generator(test_generator, steps=len(test_generator) // BATCH_SIZE, verbose=1)
    #
    # preds_cls_idx = pred.argmax(axis=-1)
    # idx_to_cls = {v: k for k, v in test_generator.class_indices.items()}
    # preds_cls = np.vectorize(idx_to_cls.get)(preds_cls_idx)
    # true_cls = test_generator.classes
    # filenames_to_cls = list(zip(test_generator.filenames, preds_cls, true_cls))
    # df = pd.DataFrame(filenames_to_cls)
    # df.to_csv(r'5_test_result.csv')
    # print(sum(preds_cls_idx[:, 0] == true_cls) / len(true_cls))
    #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

    test_generator.reset()

    pred = model.predict_generator(test_generator, steps=math.ceil(len(test_generator) / BATCH_SIZE), verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    labels = dict((v,k) for k,v in CLASSES)
    predictions = [labels[k] for k in predicted_class_indices]

    #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

    results = pd.DataFrame({"Filename": test_generator.filenames,
                            "Predictions": predictions,
                            "TRUE class": test_generator.classes})
    results.to_csv(r'2_test_result.csv')
    #

    # # 5
    # true_labels = test_generator.classes
    # preds = pred.argmax(axis=-1)
    # print("!!! FFFFFFF")
    # print(sum(preds[:, 0] == true_labels) / len(true_labels))


def plot_test_results(model: Model, evaluation):
    key2name = {'accuracy': 'Accuracy', 'loss': 'Loss',
                'val_accuracy': 'Validation Accuracy', 'val_loss': 'Validation Loss'}
    results = []
    for i, key in enumerate(model.metrics_names):
        results.append('%s = %.2f' % (key2name[key], evaluation[i]))
    print(", ".join(results))


def plot_acc(hist_path):
    plt.clf()
    with open(hist_path, 'rb') as f:
        hist = pickle.load(f)
    plt.plot(hist["accuracy"])
    plt.plot(hist["val_accuracy"])
    # key2name = {'acc':'Accuracy', 'loss':'Loss',
    #     'val_accuracy':'Validation Accuracy', 'val_loss':'Validation Loss'}
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(HIST_PLOT_PATH + 'model_accuracy.png')
    plt.show()

def save_plot_history(hist_path):
    with open(hist_path, 'rb') as f:
        hist = pickle.load(f)

    key_metrics = {'accuracy': 'Accuracy', 'loss': 'Loss',
                'val_accuracy': 'Validation Accuracy', 'val_loss': 'Validation Loss'}

    fig = plt.figure()

    metrics = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
    for i, metric in enumerate(metrics):
        trace = hist[metric]
        plt.subplot(2, 2, i + 1)
        plt.plot(range(len(trace)), trace)
        plt.title(key_metrics[metric])

    fig.set_tight_layout(True)
    fig.savefig(HIST_PLOT_PATH + '_model_metrics.png')


def save_history(hist_path, hist):
    with open(hist_path, 'wb') as f:
        pickle.dump(hist.history, f)

def get_callbacks(model_name: str, patience) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
    logdir = (LOG_PATH)  # create a folder for each model.
    tensorboard_callback = TensorBoard(log_dir=logdir)
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=patience,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=1,
        mode='max',
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    model_checkpoint_callback = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',# acc, val_acc, loss, val_loss
        verbose=1,
        save_best_only=False,  # TODO CHECK TRUE later, save the best model
        mode='max',
        save_weights_only=False,
        period=SAVE_PERIOD  # save every SAVE_PERIOD epoch
    )

    csv_logger = CSVLogger(MODEL_CSV_HIST_PATH, append=True)

    return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback, csv_logger]


import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score

from data.const import BATCH_SIZE, MODEL_NAME


def get_preds_and_labels(model, generator):
    """
    Get predictions and labels from the generator

    :param model: A Keras model object
    :param generator: A Keras ImageDataGenerator object

    :return: A tuple with two Numpy Arrays. One containing the predictions
    and one containing the labels
    """
    preds = []
    labels = []
    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
        x, y = next(generator)
        preds.append(model.predict(x))
        labels.append(y)
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()


class Metrics(Callback):
    """
    A custom Keras callback for saving the best model
    according to the Quadratic Weighted Kappa (QWK) metric
    """

    def __init__(self, model, val_generator):
        """ Save params in constructor
        """
        self.model = model
        self.val_generator = val_generator

    def on_train_begin(self, logs={}):
        """
        Initialize list of QWK scores on validation data
        """
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Gets QWK score on the validation data

        :param epoch: The current epoch number
        """
        # Get predictions and convert to integers
        y_pred, labels = get_preds_and_labels(self.model, self.val_generator)
        y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)
        # We can use sklearns implementation of QWK straight out of the box
        # as long as we specify weights as 'quadratic'
        _val_kappa = cohen_kappa_score(labels, y_pred, weights='quadratic')
        self.val_kappas.append(_val_kappa)
        print(f"val_kappa: {round(_val_kappa, 4)}")
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save(MODEL_NAME)
        return