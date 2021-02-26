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

NUM_CLASSES_TRAIN = 60
NUM_CLASSES_TEST = 60
IMG_SIZE = 380 #b4
BATCH_SIZE = 16 # TODO increase or decrease to fit your GPU
SAVE_PERIOD = 1
LEARNING_RATE=1e-2 #0.5
UNFREEZE_LEARNING_RATE=1e-4
N_EPOCHS=100
N_WORKERS=1 #0
TOP_DROPOUT_RATE=0.2
MODEL_NAME='12_effnet_b4'
HISTORY_NAME= MODEL_NAME
WEIGHTS="noisy-student"
PATIENCE=5

if platform == "linux" or platform == "linux2":
    SYS_PATH = '/home/kenny/PycharmProjects/classify_handshapes' # linux
    TRAIN_PATH = '/media/kenny/Extra/downloads/1mil/nds-v2-training/'
    TEST_PATH = '/media/kenny/Extra/downloads/1mil/ph2014-dev-set-handshape-annotations/'
elif platform == "darwin":
    SYS_PATH = '/Users/madina/PycharmProjects/classify_handshapes' # OS X
    TRAIN_PATH = '/Users/madina/Desktop/dataset/1miohands-v2-training/nds-v2-training/'
    TEST_PATH = '/Users/madina/Desktop/dataset/1miohands-v2-training/ph2014-dev-set-handshape-annotations/'
elif platform == "win32":
    SYS_PATH='' # Windows...
    TRAIN_PATH = ''
    TEST_PATH = ''

CLASSES = [str(x) for x in np.arange(1, 61, 1).tolist()]

SAVE_DIR = "models/"+MODEL_NAME
RES_DIR = "results/"+MODEL_NAME
LOG_DIR = "logs/scalars"
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

MODEL_PATH = os.path.join(SAVE_DIR, MODEL_NAME + "_model.hdf5")#"_epoch-{epoch:02d}_val_loss-{val_loss:.2f}_val_acc-{val_acc:.2f}.hdf5")
HIST_PATH = os.path.join(RES_DIR, HISTORY_NAME + ".kerashist")
HIST_PLOT_PATH = os.path.join(RES_DIR, HISTORY_NAME)
LOG_PATH = os.path.join(LOG_DIR, MODEL_NAME+ "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
PLOT_PATH = os.path.join(WORK_DIR, MODEL_NAME+ "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))