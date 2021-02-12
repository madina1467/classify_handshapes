import tensorflow as tf
from keras import layers, Model
from keras.utils import plot_model

from efficientnet.keras import EfficientNetB5
import sys


sys.path.append('/home/kenny/PycharmProjects/classify_handshapes')
from data.dataset import loadDatabase
from data.const import IMG_SIZE, NUM_CLASSES
from model_func import run_model, plot_results


def build_model(model_name) -> Model:

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # x = img_augmentation(inputs)
    model = EfficientNetB5(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    model.summary()
    plot_model(model, to_file=model_name + ".jpg", show_shapes=True)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def run():
    train_generator, validation_generator, test_generator = loadDatabase(False)

    eff_net_history = run_model(
        model_name="eff_net_b5_imagenet",
        model_function=build_model("eff_net_b5_imagenet"),
        lr=0.5, n_epochs=100, n_workers=10,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )

    plot_results(eff_net_history)

if __name__ == '__main__':
    run()


