import pandas as pd
import matplotlib.pyplot as plt
from os import path
import sys

from keras.preprocessing.image import ImageDataGenerator

sys.path.append('/home/kenny/PycharmProjects/classify_handshapes')
from data.load_labels import loadLabels
from data.const import IMG_SIZE, BATCH_SIZE


def loadDatabase(visualize=False):
    loadLabels()
    train = pd.read_csv(r'../misc/train.csv',dtype=str, index_col=[0])
    test = pd.read_csv(r'../misc/test.csv',dtype=str, index_col=[0])
    return createGenerators(train, test, visualize)


def createGenerators(train: pd.DataFrame, test: pd.DataFrame, visualize=False):

    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
        horizontal_flip=True,
        validation_split=0.25
    )  # create an ImageDataGenerator with multiple image augmentations
    validation_generator = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.25)  # except for rescaling, no augmentations are needed for validation and testing generators
    test_generator = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.25)
    # visualize image augmentations
    if visualize == True:
        visualizeAugmentations(train_generator, pd.concat([train, test]))

    train_generator = train_generator.flow_from_dataframe(
        dataframe=train,
        # directory="./train/",
        x_col="path",
        y_col="label",
        subset="training",
        # batch_size=32,
        # seed=42,
        # target_size=(32, 32),
        shuffle=True,
        class_mode="categorical",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )

    validation_generator = validation_generator.flow_from_dataframe(
        dataframe=train,
        x_col="path",
        y_col="label",
        subset="validation",
        shuffle=True,
        class_mode="categorical",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )
    test_generator = test_generator.flow_from_dataframe(
        dataframe=test,
        x_col="path",
        y_col=None,
        shuffle=False,
        class_mode=None,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )
    return train_generator, validation_generator, test_generator



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


train, val, test = loadDatabase(True)