import pandas as pd
import glob
import os
import sys

from os import path
from data.const import TRAIN_PATH, TEST_PATH, SYS_PATH, ITERATION, MODEL_NAME

sys.path.append(SYS_PATH)

def findFiles(path, rec): return glob.glob(path,recursive=rec)

def getCorrectID(value):
    keys = ['1', '2', '3', '3_hook', '4', '5', '6', '7', '8', 'a', 'b', 'b_nothumb',
            'b_thumb', 'cbaby', 'obaby', 'by', 'c', 'd', 'e', 'f', 'f_open', 'fly',
            'fly_nothumb', 'g', 'h', 'h_hook', 'h_thumb', 'i', 'jesus', 'k', 'l_hook',
            'middle', 'm', 'n', 'o', 'index', 'index_flex', 'index_hook', 'pincet',
            'ital', 'ital_thumb', 'ital_nothumb', 'ital_open', 'r', 's', 'write',
            'spoon', 't', 'v', 'v_flex', 'v_hook', 'v_thumb', 'w', 'y', 'ae',
            'ae_thumb', 'pincet_double', 'obaby_double', 'm2', 'jesus_thumb']
    return keys.index(value) + 1

# def steps_needed(csv_files):
#     steps = 0
#
#     for file in csv_files:
#     df = pd.read_csv(file, chunksize=self.chunk_size)
#     for df_chunk in df:
#         chunk_steps = math.ceil(len(df_chunk) / self.batch_size)
#         steps += chunk_steps
#     return steps


def getTeacherAnnotations():
    if path.exists("../misc/all_train.csv") and path.exists("../misc/all_test.csv"):
        train = pd.read_csv(r'../misc/all_train.csv', index_col=[0])
        test = pd.read_csv(r'../misc/all_test.csv', index_col=[0])
    else:
        print('getAllAnnotations(): Start reading train/test files')
        train = pd.read_csv(TRAIN_PATH + '1miohands-v2-trainingalignment.txt', sep=" ", header=None, names=["path", "label"])
        train.path = train.path.apply(
            lambda x: x.replace('/work/cv2/koller/features/danish_nz_ph2014/hand.20151016/data/joint/',
                                TRAIN_PATH))

        test = pd.read_csv(TEST_PATH + '3359-ph2014-MS-handshape-annotations.txt', sep=" ", header=None, names=["path", "label"])
        test.path = test.path.apply(lambda x: x.replace('images/', TEST_PATH))
        test.label = test.label.apply(lambda x: getCorrectID(x))

        print('getAllAnnotations(): End reading train/test files')

        train.to_csv(r'../misc/all_train.csv')
        test.to_csv(r'../misc/all_test.csv')
    return train, test


def getStudentAnnotations():
    # if path.exists("../misc/student_train.csv") and path.exists("../misc/test.csv"):
    #     train = pd.read_csv(r'../misc/student_train.csv', index_col=[0])
    #     test = pd.read_csv(r'../misc/test.csv', index_col=[0])
    # else:
    print('getStudentAnnotations(): Start reading train/test files')
    file = './train/results/labeling/' + ITERATION + '_' + MODEL_NAME + '_teacher_unlabeled_result.csv'
    # train = pd.read_csv(file)

    train = pd.read_csv(file, sep=",", header=0, names=["Filename", "Prediction", "PredPercent", "Prediction2", "PredPercent2", "Prediction3", "PredPercent3"])
    train['PredPercent'] = pd.to_numeric(train['PredPercent'], downcast="float", errors='coerce')
    train.drop(train[train.PredPercent < 0.3].index, inplace=True)
    train.rename(columns={'Filename': 'path', 'Prediction': 'label'}, inplace=True)
    train.drop(["PredPercent", "Prediction2", "PredPercent2", "Prediction3", "PredPercent3"], axis=1, inplace=True)

    test = pd.read_csv(r'./misc/test.csv', index_col=[0])

    print('getStudentAnnotations(): End reading train/test files')

    train.to_csv(r'./misc/student_train.csv')
        # test.to_csv(r'../misc/test.csv')
    return train, test

def getLabelFromDF(df, path):
    temp = df[df.path == path]
    if temp.size == 0:
        print('!!NF', path)
        return -1
    return temp.label.iat[0]


def createLabels(all_labels, expected_dir):
    df = pd.DataFrame(columns=['path', 'label', 'name'])
    counter_not_found = 0
    for file_name_relative in findFiles(expected_dir + "**/*.png", True):
        label = getLabelFromDF(all_labels, file_name_relative)
        if label == -1:
            counter_not_found = counter_not_found + 1
        else:
            file_name_absolute = os.path.basename(file_name_relative)
            df = df.append({'path': file_name_relative, 'label': label, 'name': file_name_absolute}, ignore_index=True)
    return df, counter_not_found

def loadTeacherLabels():
    if path.exists("../misc/train.csv") and path.exists("../misc/test.csv"):
        train = pd.read_csv(r'../misc/train.csv', index_col=[0])
        test = pd.read_csv(r'../misc/test.csv', index_col=[0])
    else:
        print('getLabels(): Starting creating new train/test files')
        all_train, all_test = getTeacherAnnotations()

        train, nf_train = createLabels(all_train, TRAIN_PATH)
        test, nf_test = createLabels(all_test, TEST_PATH)

        print('getLabels(): End of creating new train/test files')

        train.to_csv(r'../misc/train.csv')
        test.to_csv(r'../misc/test.csv')

        print('!!NF TRAINTRAIN', nf_train)
        print('!!NF TESTTEST', nf_test)

    # return train, test

def loadStudentLabels():
    # if path.exists("../misc/student_train.csv") and path.exists("../misc/test.csv"):
    #     train = pd.read_csv(r'../misc/student_train.csv', index_col=[0])
    #     test = pd.read_csv(r'../misc/test.csv', index_col=[0])
    # else:
    print('loadStudentLabels(): Starting creating new train/test files')
    train, test = getStudentAnnotations()

    # print(train)
    return train, test

def checkTeacherClasses():
    train = pd.read_csv(r'../misc/old/train.csv', index_col=[0])
    test = pd.read_csv(r'../misc/old/test.csv', index_col=[0])

    print('!!AA train', train['label'].value_counts(ascending=True).sort_index())
    print('!!BB train', train['label'].nunique())

    print('!!AA test', test['label'].value_counts(ascending=True).sort_index())
    print('!!BB test', test['label'].nunique())

# checkClasses()
# loadLabels()
# loadStudentLabels()