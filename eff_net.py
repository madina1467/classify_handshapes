from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import EfficientNetB0

from data.const import IMG_SIZE
from data.dataset import loadDatabase
from train.train import run_model, plot_results


def adapt_efficient_net() -> Model:
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    outputs = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')(inputs)
    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")

    return model

def run():
    train_generator, validation_generator, test_generator = loadDatabase()

    eff_net_history = run_model(
        model_name="eff_net",
        model_function=adapt_efficient_net(),
        lr=0.5, n_epochs=100, n_workers=6, # TODO adjust this according to the number of CPU cores of your machine
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )

    plot_results(eff_net_history)

