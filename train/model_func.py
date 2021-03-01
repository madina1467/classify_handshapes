import math
import matplotlib.pyplot as plt
import sys
import dill as pickle
import numpy as np
import pandas as pd

from typing import Iterator, List, Union
from keras import Model
from keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model

from data.const import BATCH_SIZE, SAVE_PERIOD, MODEL_PATH, LOG_PATH, PLOT_PATH, CLASSES, HIST_PLOT_PATH, SYS_PATH, \
    MODEL_NAME, ITERATION

sys.path.append(SYS_PATH)

def run_model(
        model_name: str,
        hist_path:str,
        model_function: Model,
        n_epochs, n_workers, patience,
        train_generator: Iterator,
        validation_generator: Iterator,
        test_generator: Iterator,
) -> History:
    callbacks = get_callbacks(model_name, patience)
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

    save_history(hist_path, history)
    save_plot_history(hist_path)
    plot_acc(hist_path)

    return history  # type: ignore

def teacher_predict_unlabeled(file_name, unlabeled_generator: Iterator):
    model = load_model('models/'+file_name)

    unlabeled_generator.reset()

    pred = model.predict_generator(unlabeled_generator, steps=len(unlabeled_generator), verbose=1)
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
    key2name = {'acc': 'Accuracy', 'loss': 'Loss',
                'val_acc': 'Validation Accuracy', 'val_loss': 'Validation Loss'}
    results = []
    for i, key in enumerate(model.metrics_names):
        results.append('%s = %.2f' % (key2name[key], evaluation[i]))
    print(", ".join(results))


def plot_acc(hist_path):
    with open(hist_path, 'rb') as f:
        hist = pickle.load(f)
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    # key2name = {'acc':'Accuracy', 'loss':'Loss',
    #     'val_acc':'Validation Accuracy', 'val_loss':'Validation Loss'}
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(HIST_PLOT_PATH + 'model_accuracy.png')
    plt.show()

def save_plot_history(hist_path):
    with open(hist_path, 'rb') as f:
        hist = pickle.load(f)

    key_metrics = {'acc': 'Accuracy', 'loss': 'Loss',
                'val_acc': 'Validation Accuracy', 'val_loss': 'Validation Loss'}

    fig = plt.figure()

    metrics = ['acc', 'loss', 'val_acc', 'val_loss']
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
        monitor='val_acc',
        patience=patience,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=1,
        mode='max',
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    model_checkpoint_callback = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_acc',# acc, val_acc, loss, val_loss
        verbose=1,
        save_best_only=True,  # TODO CHECK TRUE later, save the best model
        mode='max',
        save_weights_only=False,
        period=SAVE_PERIOD  # save every SAVE_PERIOD epoch
    )
    return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback]
