TRAIN_PATH = '/media/kenny/Extra/downloads/1mil/nds-v2-training/'
TEST_PATH = '/media/kenny/Extra/downloads/1mil/ph2014-dev-set-handshape-annotations/'
NUM_CLASSES_TRAIN = 60
NUM_CLASSES_TEST = 60
IMG_SIZE = 380#b4   b5-456 # input shapes of the images should always be 224x224x3 with EfficientNetB0
BATCH_SIZE = 16 # TODO increase or decrease to fit your GPU
SAVE_PERIOD = 1
LEARNING_RATE=1e-2#0.5
N_EPOCHS=100
N_WORKERS=1#0
TOP_DROPOUT_RATE=0.2
MODEL_NAME='4_eff_net_b4_imagenet'