import pandas as pd
import matplotlib.pyplot as plt
import sys
import splitfolders
import os
from os import path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data.const import IMG_SIZE, BATCH_SIZE, CLASSES, SYS_PATH, STUDENT_ANNOTATIONS, NEW_DATASET_ANNOTATIONS
from data.randaugment import Rand_Augment

from PIL import Image
import numpy as np

sys.path.append(SYS_PATH)

def loadTeacherDatabase():
    # loadLabels() #TODO #FIX
    train = pd.read_csv(r'../misc/train.csv', dtype=str, index_col=[0])
    test = pd.read_csv(r'../misc/test.csv', dtype=str, index_col=[0])

    train_labeled = train[train.label != 0]
    train_unlabeled = train[train.label == 0]

    return createGenerators()

def loadStudentDatabase():
    # loadLabels() #TODO #FIX
    train = pd.read_csv(r'../misc/'+STUDENT_ANNOTATIONS, dtype=str, index_col=[0])
    test = pd.read_csv(r'../misc/test.csv', dtype=str, index_col=[0])

    return createGenerators()


def loadNewDatabase():
    # loadLabels() #TODO #FIX
    # train = pd.read_csv(r'../misc/train'+NEW_DATASET_ANNOTATIONS, dtype=str, index_col=[0])
    # test = pd.read_csv(r'../misc/test.csv', dtype=str, index_col=[0])
    #
    # train.drop(train.filter(regex="Unname"), axis=1, inplace=True)
    return createGenerators()

def loadDatabaseUnlabeled():
    train = pd.read_csv(r'../misc/train.csv', dtype=str, index_col=[0])
    train_unlabeled = train[train.label == '0']
    return createTestGenerator(train_unlabeled, False, False)

def loadTESTDatabase(tt = r'../misc/test.csv'):
    test = pd.read_csv(tt, dtype=str, index_col=[0])

    return createTESTGenerators(test)

def loadTESTDatabase2(n=10):
    test = pd.read_csv(r'../misc/test.csv', dtype=str, index_col=[0], nrows=n)

    return createTESTGenerators(test)

def createTestGenerator(test: pd.DataFrame, shuffle=False, to_fit=False):
    generator = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = generator.flow_from_dataframe(
        dataframe=test,
        x_col="path",
        y_col=None,
        # y_col="label",
        shuffle=shuffle,
        # class_mode="categorical",
        class_mode=None,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        to_fit=to_fit
    )
    return test_generator


randaugment = Rand_Augment()
def preprocessing_function(image):

    image = Image.fromarray(image.astype(np.uint8))
    image = np.array(randaugment(image))
    return image.astype(np.float64)

def createGenerators():

    # main_dir = '/media/kenny/Extra/downloads/1mil/train_by_columns_new'
    # output_dir = '/media/kenny/Extra/downloads/1mil/train_by_columns_new_output'

    main_dir = '/media/kenny/Extra/downloads/1mil/train_cat_cols2/1/1_1'
    output_dir = '/media/kenny/Extra/downloads/1mil/train_cat_cols2/1/1_1_output'
    splitfolders.ratio(main_dir, output=output_dir, seed=1337, ratio=(.8, .2))


    train_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.75, 1),
    shear_range=0.1,
    zoom_range=[0.75, 1],
    horizontal_flip=True,
    )
    validation_generator = ImageDataGenerator(rescale=1.0 / 255)  # except for rescaling, no augmentations are needed for validation and testing generators

    train_generator = train_generator.flow_from_directory(
        os.path.join(output_dir,'train'),
        shuffle=True,
        class_mode="categorical",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    validation_generator = validation_generator.flow_from_directory(
        os.path.join(output_dir,'val'),
        shuffle=True,
        class_mode="categorical",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    return train_generator, validation_generator


def createTESTGenerators(test: pd.DataFrame):
    test_generator = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_generator.flow_from_dataframe(
        dataframe=test,
        x_col="path",
        # y_col=None,
        y_col="label",
        shuffle=False,
        class_mode="categorical",
        # class_mode=None,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        to_fit=False
    )
    return test_generator




def visualizeAugmentations(data_generator: ImageDataGenerator, df: pd.DataFrame):
    """Visualizes the keras augmentations with matplotlib in 3x3 grid. This function is part of create_generators() and
    can be accessed from there.

    Parameters
    ----------
    data_generator : Iterator
        The keras data generator of your training data.
    df : pd.DataFrame
        The Pandas DataFrame containing your training data.
    """
    # super hacky way of creating a small dataframe with one image
    series = df.iloc[2]

    print("!!AA")
    print(series)

    # for index, row in series.iterrows():
    #     print(row['path'], row['label'], path.exists(row['path']))

    print(series['path'], series['label'], path.exists(series['path']))

    df_augmentation_visualization = pd.concat([series, series], axis=1).transpose()

    iterator_visualizations = data_generator.flow_from_dataframe(  # type: ignore
        dataframe=df_augmentation_visualization,
        x_col="path",
        y_col="label",
        # class_mode="raw",
        target_size=(IMG_SIZE, IMG_SIZE),  # size of the image
        batch_size=1,  # use only one image for visualization
    )

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)  # create a 3x3 grid
        batch = next(iterator_visualizations)  # get the next image of the generator (always the same image)
        img = batch[0]  # type: ignore
        print('!!!! img:', img)
        print('!!!! img.shape:', img.shape)
        img = img[0, :, :, :]  # remove one dimension for plotting without issues
        plt.imshow(img)
    plt.show()
    plt.close()


# train, val, b5 = loadDatabase(True)