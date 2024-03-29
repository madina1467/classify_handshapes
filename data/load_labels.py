import pandas as pd
import glob
import os
import sys

from os import path
from data.const import TRAIN_PATH, TEST_PATH, SYS_PATH, ITERATION, MODEL_NAME, LAST_TEACHER_MODEL_NAME, \
    NEW_DATASET_ANNOTATIONS

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

def createNewLabels(path = '/media/kenny/Extra/downloads/1mil/train_by_columns'):
    df = pd.DataFrame(columns=['path', 'label'])

    for root, d_names, f_names in os.walk(path):
        for f in f_names:
            df = df.append({'path': os.path.join(root, f), 'label': root.rsplit('/', 1)[-1]}, ignore_index=True)
    df = df.reset_index()
    df = df.drop(['index'], axis=1)

    return df

def createAnnotation():
    df = createNewLabels()
    print('createAnnotation(): Start reading createAnnotation createAnnotation createAnnotation')
    print(df)
    df.to_csv(r'../misc/train'+NEW_DATASET_ANNOTATIONS)


    test = pd.read_csv(TEST_PATH + '3359-ph2014-MS-handshape-annotations.txt', sep=" ", header=None,
                       names=["path", "label"])
    test.path = test.path.apply(lambda x: x.replace('images/', TEST_PATH))
    test.label = test.label.apply(lambda x: getCorrectID(x))

    print('getAllAnnotations(): End reading train/b5 files')
    test.to_csv(r'../misc/test'+NEW_DATASET_ANNOTATIONS)




def getAnnotations():
    if path.exists("../misc/new_train2.csv") and path.exists("../misc/all_test.csv"):
        train = pd.read_csv(r'../misc/new_train2.csv', index_col=[0])
        test = pd.read_csv(r'../misc/all_test2.csv', index_col=[0])
    else:
        print('getAllAnnotations(): Start reading train/b5 files AAAAAAAAAAAAAA')
        train = pd.read_csv(TRAIN_PATH + 'new_dataset22.csv', sep=",", header=None, names=["path", "label"])
        train.path = train.path.apply(
            lambda x: x.replace('train_by_category/', # train_by_classes
                                TRAIN_PATH))

        test = pd.read_csv(TEST_PATH + '3359-ph2014-MS-handshape-annotations.txt', sep=" ", header=None, names=["path", "label"])
        test.path = test.path.apply(lambda x: x.replace('images/', TEST_PATH))
        test.label = test.label.apply(lambda x: getCorrectID(x))

        print('getAllAnnotations(): End reading train/b5 files')

        train.to_csv(r'../misc/new_train2.csv')
        test.to_csv(r'../misc/all_test2.csv')
    return train, test


def getTeacherAnnotations():
    if path.exists("../misc/all_train.csv") and path.exists("../misc/all_test.csv"):
        train = pd.read_csv(r'../misc/all_train.csv', index_col=[0])
        test = pd.read_csv(r'../misc/all_test.csv', index_col=[0])
    else:
        print('getAllAnnotations(): Start reading train/b5 files')
        train = pd.read_csv(TRAIN_PATH + '1miohands-v2-trainingalignment.txt', sep=" ", header=None, names=["path", "label"])
        train.path = train.path.apply(
            lambda x: x.replace('/work/cv2/koller/features/danish_nz_ph2014/hand.20151016/data/joint/',
                                TRAIN_PATH))

        test = pd.read_csv(TEST_PATH + '3359-ph2014-MS-handshape-annotations.txt', sep=" ", header=None, names=["path", "label"])
        test.path = test.path.apply(lambda x: x.replace('images/', TEST_PATH))
        test.label = test.label.apply(lambda x: getCorrectID(x))

        print('getAllAnnotations(): End reading train/b5 files')

        train.to_csv(r'../misc/all_train.csv')
        test.to_csv(r'../misc/all_test.csv')
    return train, test


def getStudentAnnotations():
    # if path.exists("../misc/student_train.csv") and path.exists("../misc/b5.csv"):
    #     train = pd.read_csv(r'../misc/student_train.csv', index_col=[0])
    #     b5 = pd.read_csv(r'../misc/b5.csv', index_col=[0])
    # else:
    print('getStudentAnnotations(): Start reading train/b5 files')
    file = 'train/results/labeling/' + ITERATION + '_' + LAST_TEACHER_MODEL_NAME + '_teacher_unlabeled_result.csv'
    # train = pd.read_csv(file)

    train = pd.read_csv(os.path.join(SYS_PATH, file), sep=",", header=0, names=["Filename", "Prediction", "PredPercent", "Prediction2", "PredPercent2", "Prediction3", "PredPercent3"])
    train['PredPercent'] = pd.to_numeric(train['PredPercent'], downcast="float", errors='coerce')

    print(train.shape)
    train.drop(train[train.PredPercent < 0.7].index, inplace=True)
    print(train.shape)
    train.rename(columns={'Filename': 'path', 'Prediction': 'label'}, inplace=True)
    train.label = train.label + 1
    train.drop(["PredPercent", "Prediction2", "PredPercent2", "Prediction3", "PredPercent3"], axis=1, inplace=True)

    test = pd.read_csv(os.path.join(SYS_PATH, 'misc/b5.csv'), index_col=[0])

    print('getStudentAnnotations(): End reading train/b5 files')
    print(train.shape)
    train.to_csv(os.path.join(SYS_PATH, 'misc/student_train_drop_if<0.7.csv'))
        # b5.to_csv(r'../misc/b5.csv')
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
        print('getLabels(): Starting creating new train/b5 files')
        all_train, all_test = getTeacherAnnotations()

        train, nf_train = createLabels(all_train, TRAIN_PATH)
        test, nf_test = createLabels(all_test, TEST_PATH)

        print('getLabels(): End of creating new train/b5 files')

        train.to_csv(r'../misc/train.csv')
        test.to_csv(r'../misc/b5.csv')

        print('!!NF TRAINTRAIN', nf_train)
        print('!!NF TESTTEST', nf_test)

    # return train, b5

def loadStudentLabels():
    # if path.exists("../misc/student_train.csv") and path.exists("../misc/b5.csv"):
    #     train = pd.read_csv(r'../misc/student_train.csv', index_col=[0])
    #     b5 = pd.read_csv(r'../misc/b5.csv', index_col=[0])
    # else:
    print('loadStudentLabels(): Starting creating new train/b5 files')
    train, test = getStudentAnnotations()

    # print(train)
    return train, test

def checkTeacherClasses():
    train = pd.read_csv(r'../misc/old/train.csv', index_col=[0])
    test = pd.read_csv(r'../misc/old/test.csv', index_col=[0])

    print('!!AA train', train['label'].value_counts(ascending=True).sort_index())
    print('!!BB train', train['label'].nunique())

    print('!!AA b5', test['label'].value_counts(ascending=True).sort_index())
    print('!!BB b5', test['label'].nunique())

# checkClasses()
# loadLabels()
# loadStudentLabels()


if __name__ == '__main__':
    print('AA')
    # getAnnotations()

    createAnnotation()