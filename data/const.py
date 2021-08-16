import numpy as np
import os

from os import path
from datetime import datetime
from sys import platform
#
# | Base model | resolution|
# |----------------|-----|
# | EfficientNetB0 | 224 |
# | EfficientNetB1 | 240 |
# | EfficientNetB2 | 260 |
# | EfficientNetB3 | 300 |
# | EfficientNetB4 | 380 |
# | EfficientNetB5 | 456 |
# | EfficientNetB6 | 528 |
# | EfficientNetB7 | 600 |

NUM_CLASSES_TRAIN = 9
NUM_CLASSES_TEST = 9
IMG_SIZE = 456
BATCH_SIZE = 8 # TODO increase or decrease to fit your GPU
SAVE_PERIOD = 1
LEARNING_RATE=1e-2 # 0.01 #0.5
UNFREEZE_LEARNING_RATE=1e-4
N_EPOCHS=100
N_WORKERS=0 #0
TOP_DROPOUT_RATE=0.2
LAST_TEACHER_MODEL_NAME='31_b5_RMSprop_f28_22_f24_08'
MODEL_NAME='31_b5_RMSprop_f28_22_f24_08'
HISTORY_NAME= MODEL_NAME+'HISTORY'
WEIGHTS="noisy-student"
PATIENCE=80
N_EPOCHS_UNFREEZE=20
ITERATION='1'
STUDENT_ANNOTATIONS='student_train_drop_if<0.7.csv'
NEW_DATASET_ANNOTATIONS='dataset_labels.csv'#'new_train2.csv'

if platform == "linux" or platform == "linux2":
    SYS_PATH = '/home/kenny/PycharmProjects/classify_handshapes' # linux
    TRAIN_PATH = '/media/kenny/Extra/downloads/1mil/train_by_category/'#train_by_classes/' #nds-v2-training/'
    TEST_PATH = '/media/kenny/Extra/downloads/1mil/ph2014-dev-set-handshape-annotations/'
    LABELS_PATH = '/home/kenny/PycharmProjects/classify_handshapes/train/results/labeling/'
elif platform == "darwin":
    SYS_PATH = '/Users/madina/PycharmProjects/classify_handshapes' # OS X
    TRAIN_PATH = '/Users/madina/Desktop/dataset/1miohands-v2-training/nds-v2-training/'
    TEST_PATH = '/Users/madina/Desktop/dataset/1miohands-v2-training/ph2014-dev-set-handshape-annotations/'
elif platform == "win32":
    SYS_PATH='' # Windows...
    TRAIN_PATH = ''
    TEST_PATH = ''

CLASSES = [str(x) for x in np.arange(1, NUM_CLASSES_TRAIN, 1).tolist()]

SAVE_DIR = "/media/kenny/Extra/models/"+MODEL_NAME
RES_DIR = "/media/kenny/Extra/results/"+MODEL_NAME
LABELING_DIR = "results/labeling/"
LOG_DIR = "logs/scalars/"
WORK_DIR = "model_architecture"

if ~path.exists(SAVE_DIR):
    try:
        os.mkdir(SAVE_DIR)
    except OSError:
        print ("Creation of the directory %s failed" % SAVE_DIR)
    else:
        print ("Successfully created the directory %s " % SAVE_DIR)

if ~path.exists(RES_DIR):
    try:
        os.mkdir(RES_DIR)
    except OSError:
        print ("Creation of the directory %s failed" % RES_DIR)
    else:
        print ("Successfully created the directory %s " % RES_DIR)

# MODEL_PATH = os.path.join(SAVE_DIR, MODEL_NAME + "_model.hdf5")#"_epoch-{epoch:02d}_val_loss-{val_loss:.2f}_val_acc-{val_accuracy:.2f}.hdf5")
MODEL_PATH = os.path.join(SAVE_DIR, MODEL_NAME + "_epoch-{epoch:02d}_val_loss-{val_loss:.2f}_val_acc-{val_accuracy:.2f}.hdf5")
MODEL_CSV_HIST_PATH = os.path.join(RES_DIR, HISTORY_NAME + "_log.csv")
HIST_PATH = os.path.join(RES_DIR, HISTORY_NAME + ".kerashist")
HIST_PLOT_PATH = os.path.join(RES_DIR, HISTORY_NAME)
LOG_PATH = os.path.join(LOG_DIR, MODEL_NAME+ "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
PLOT_PATH = os.path.join(WORK_DIR, MODEL_NAME+ "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))