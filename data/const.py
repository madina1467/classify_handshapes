from datetime import datetime

TRAIN_PATH = '/media/kenny/Extra/downloads/1mil/nds-v2-training/'
TEST_PATH = '/media/kenny/Extra/downloads/1mil/ph2014-dev-set-handshape-annotations/'
NUM_CLASSES_TRAIN = 60
NUM_CLASSES_TEST = 60
IMG_SIZE = 380#b4   b5-456 # input shapes of the images should always be 224x224x3 with EfficientNetB0
BATCH_SIZE = 16 # TODO increase or decrease to fit your GPU
SAVE_PERIOD = 1
LEARNING_RATE=1e-2#0.5
N_EPOCHS=3
N_WORKERS=1#0
TOP_DROPOUT_RATE=0.2
MODEL_NAME='5_eff_net_b4_imagenet'
HISTORY_NAME= 'HISTORY_' + MODEL_NAME

import os
SAVE_DIR = "models"
RES_DIR = "results"
LOG_DIR = "logs/scalars"
WORK_DIR = "work"
MODEL_PATH = os.path.join(RES_DIR, MODEL_NAME + "_weights_epoch-{epoch:02d}_val_loss-{val_loss:.2f}_val_acc-{val_acc:.2f}.hdf5")
HIST_PATH = os.path.join(RES_DIR, HISTORY_NAME + ".kerashist")
LOG_PATH = os.path.join(LOG_DIR, MODEL_NAME+ "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
PLOT_PATH = os.path.join(WORK_DIR, MODEL_NAME+ "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))