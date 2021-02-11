import pandas as pd
import glob
import os
from os import path

from const import TRAIN_PATH, TEST_PATH

def findFiles(path, rec=False): return glob.glob(path,recursive=rec)

def get_id(value):
    keys = ['1', '2', '3', '3_hook', '4', '5', '6', '7', '8', 'a', 'b', 'b_nothumb',
            'b_thumb', 'cbaby', 'obaby', 'by', 'c', 'd', 'e', 'f', 'f_open', 'fly',
            'fly_nothumb', 'g', 'h', 'h_hook', 'h_thumb', 'i', 'jesus', 'k', 'l_hook',
            'middle', 'm', 'n', 'o', 'index', 'index_flex', 'index_hook', 'pincet',
            'ital', 'ital_thumb', 'ital_nothumb', 'ital_open', 'r', 's', 'write',
            'spoon', 't', 'v', 'v_flex', 'v_hook', 'v_thumb', 'w', 'y', 'ae',
            'ae_thumb', 'pincet_double', 'obaby_double', 'm2', 'jesus_thumb']
    return keys.index(value) + 1


def getAllAnnotations():
    if path.exists("../misc/all_train.csv") and path.exists("../misc/all_test.csv"):
        train = pd.read_csv(r'../misc/all_train.csv', index_col=[0])
        test = pd.read_csv(r'../misc/all_test.csv', index_col=[0])
    else:
        train = pd.read_csv(TRAIN_PATH + '1miohands-v2-trainingalignment.txt', sep=" ", header=None, names=["path", "label"])
        train.path = train.path.apply(
            lambda x: x.replace('/work/cv2/koller/features/danish_nz_ph2014/hand.20151016/data/joint/',
                                TRAIN_PATH))

        test = pd.read_csv(TEST_PATH + '3359-ph2014-MS-handshape-annotations.txt', sep=" ", header=None, names=["path", "label"])
        test.path = test.path.apply(lambda x: x.replace('images/', TEST_PATH))
        test.label = test.label.apply(lambda x: get_id(x))

        train.to_csv(r'../misc/all_train.csv')
        test.to_csv(r'../misc/all_test.csv')
    return train, test

def getCorrectLabel(df, path):
    print('!!AAA', path)
    if ~(path in df.index):
        print('!!NF', path)
        return -1
    temp = df.loc[path].label
    print('!!BBB', temp)
    return temp.iat[0]


    # if temp.size == 0:
    #     print('!!NF', path)
    #     return -1
    # else:
    #     return temp.iat[0]

    # print('!!!BBB ', df[df.path.str.contains(path)].label)
    # print('!!!CCC ', df[df.path.str.contains(path)].label.iat[0])
    # print('!!!DDD ', df[df.path.str.contains(path)])
    # return df[df.path.str.contains(path)].label.iat[0]


def createLabels(allLabels, expectedDir):
    df = pd.DataFrame(columns=['path', 'label', 'name'])
    counter_not_found = 0
    for file_name_relative in findFiles(expectedDir + "**/*.png", True):
        label = getCorrectLabel(allLabels, file_name_relative)
        if label == -1:
            counter_not_found =+ 1
            print('!!CCC +1 ', counter_not_found)
        else:
            file_name_absolute = os.path.basename(file_name_relative)
            df = df.append({'path': file_name_relative, 'label': label, 'name': file_name_absolute}, ignore_index=True)
    return df, counter_not_found

def getLabels():
    if path.exists("../misc/train.csv") and path.exists("../misc/test.csv"):
        train = pd.read_csv(r'../misc/train.csv', index_col=[0])
        test = pd.read_csv(r'../misc/test.csv', index_col=[0])
    else:
        allTrain, allTest = getAllAnnotations()
        allTrain.index.names = ['path']
        allTest.index.names = ['path']

        # trainExpectedDir = '/Users/madina/Desktop/dataset/1miohands-v2-training/danish_nz_ph2014/'
        # testExpectedDir = '/Users/madina/Desktop/dataset/1miohands-v2-training/final_phoenix_noPause_noCompound_lefthandtag_noClean'

        train, nf_train = createLabels(allTrain, TRAIN_PATH)
        test, nf_test = createLabels(allTest, TEST_PATH)

        train.to_csv(r'../misc/train.csv')
        test.to_csv(r'../misc/test.csv')

        print('!!NF TRAINTRAIN', nf_train)
        print('!!NF TESTTEST', nf_test)

    return train, test

# getLabels()

df = pd.read_csv(r'../misc/all_test.csv', index_col=[0])
# df.index.names = ['path']
# df = df.set_index('path')
path = '/media/kenny/Extra/downloads/1mil/ph2014-dev-set-handshape-annotations/final_phoenix_noPause_noCompound_lefthandtag_noClean/01February_2011_Tuesday_heute_default-8/1/*.png_fn000030-0.png'
# if ~(path in df.index):
#     print('!!NF', path)
# else:
#     temp = df.loc[path].label
#     print('!!BBB', temp)
# # return temp.iat[0]
#
# data_top = df.head(5)
# for row in data_top.index:
#     print(row, end="\n\n ")
# data["Indexes"]=
# print(df["path"].str.find(path))
# print(df.filter(like=path, axis=0))
print(df[df.path == 'path'])














#
# def createLabels():
#
#     if path.exists("train.csv") and path.exists("test.csv"):
#         train = pd.read_csv(r'train.csv', index_col=[0])
#         test = pd.read_csv(r'test.csv', index_col=[0])
#     else:
#         train = pd.read_csv(TRAIN_PATH + '1miohands-v2-trainingalignment.txt', sep=" ", header=None, names=["path", "label"])
#         train.path = train.path.apply(
#             lambda x: x.replace('/work/cv2/koller/features/danish_nz_ph2014/hand.20151016/data/joint/', TRAIN_PATH))
#         for index, row in train.iterrows():
#             if not (path.exists(row['path'])):
#                 train.drop(index, inplace=True)
#
#         test = pd.read_csv(TEST_PATH + '3359-ph2014-MS-handshape-annotations.txt', sep=" ", header=None, names=["path", "label"])
#         test.path = test.path.apply(lambda x: x.replace('images/', TEST_PATH))
#         for index, row in test.iterrows():
#             if not (path.exists(row['path'])):
#                 test.drop(index, inplace=True)
#         test.label = test.label.apply(lambda x: get_id(x))
#
#         train.to_csv(r'train.csv')
#         test.to_csv( r'test.csv')
#
#     return train[['path', 'label']].astype(str), test[['path', 'label']].astype(str)

# train_labels, test_labels = createLabels()