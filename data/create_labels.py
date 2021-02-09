
import glob
import pandas as pd
from os import path

from const import TRAIN_PATH, TEST_PATH


def findFiles(path): return glob.glob(path)

def get_id(value):
    keys = ['1', '2', '3', '3_hook', '4', '5', '6', '7', '8', 'a', 'b', 'b_nothumb',
            'b_thumb', 'cbaby', 'obaby', 'by', 'c', 'd', 'e', 'f', 'f_open', 'fly',
            'fly_nothumb', 'g', 'h', 'h_hook', 'h_thumb', 'i', 'jesus', 'k', 'l_hook',
            'middle', 'm', 'n', 'o', 'index', 'index_flex', 'index_hook', 'pincet',
            'ital', 'ital_thumb', 'ital_nothumb', 'ital_open', 'r', 's', 'write',
            'spoon', 't', 'v', 'v_flex', 'v_hook', 'v_thumb', 'w', 'y', 'ae',
            'ae_thumb', 'pincet_double', 'obaby_double', 'm2', 'jesus_thumb']
    return keys.index(value) + 1


def createLabels():

    if path.exists("train.csv") and path.exists("test.csv"):
        train = pd.read_csv(r'train.csv', index_col=[0])
        test = pd.read_csv(r'test.csv', index_col=[0])
    else:
        train = pd.read_csv(TRAIN_PATH + '1miohands-v2-trainingalignment.txt', sep=" ", header=None, names=["path", "label"])
        train.path = train.path.apply(
            lambda x: x.replace('/work/cv2/koller/features/danish_nz_ph2014/hand.20151016/data/joint/', TRAIN_PATH))
        for index, row in train.iterrows():
            if not (path.exists(row['path'])):
                train.drop(index, inplace=True)

        test = pd.read_csv(TEST_PATH + '3359-ph2014-MS-handshape-annotations.txt', sep=" ", header=None, names=["path", "label"])
        test.path = test.path.apply(lambda x: x.replace('images/', TEST_PATH))
        for index, row in test.iterrows():
            if not (path.exists(row['path'])):
                test.drop(index, inplace=True)
        test.label = test.label.apply(lambda x: get_id(x))

        train.to_csv(r'train.csv')
        test.to_csv( r'test.csv')

    return train[['path', 'label']].astype(str), test[['path', 'label']].astype(str)

# train_labels, test_labels = createLabels()