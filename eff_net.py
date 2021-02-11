import tensorflow as tf
from tensorflow.keras import layers, models, Model
from efficientnet.keras import EfficientNetB5
from tensorflow.keras.utils import plot_model

from data.const import IMG_SIZE, NUM_CLASSES
from data.dataset import loadDatabase
from train.train import run_model, plot_results


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
    train_generator, validation_generator, test_generator = loadDatabase()

    eff_net_history = run_model(
        model_name="eff_net_b5",
        model_function=build_model("eff_net_b5"),
        lr=0.5, n_epochs=100, n_workers=10,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )

    plot_results(eff_net_history)
