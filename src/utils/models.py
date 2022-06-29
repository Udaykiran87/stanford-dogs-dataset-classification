import tensorflow as tf
import logging
from src.utils import get_timestamp
import os

def get_inceptionV3_model(input_shape, model_path):
    model = tf.keras.applications.InceptionV3(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False
    )

    model.save(model_path)
    logging.info(f"Inception_V3 base model is saved at: {model_path}")
    return model

def prepare_model(model, CLASSES, freeze_all, freeze_till, learning_rate):
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 0):
        for layer in model.layers[:-freeze_till]:
            layer.trainable = False

    ## add our fully connected layers
    pool_out = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    drop_out = tf.keras.layers.Dropout(0.2)(pool_out)
    prediction = tf.keras.layers.Dense(
        units = CLASSES,
        activation = "softmax"
    )(drop_out)

    full_model = tf.keras.models.Model(
        inputs = model.input,
        outputs = prediction
    )

    full_model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ["accuracy"],
    )

    logging.info("custom model is compiled and ready to be trained")

    return full_model


def load_full_model(untrained_full_model_path):
    model = tf.keras.models.load_model(untrained_full_model_path)
    logging.info(f"untrained model is read from: {untrained_full_model_path}")
    return model

def get_unique_path_to_save_model(trained_model_dir, model_name="model"):
    timestamp = get_timestamp(model_name)
    unique_model_name = f"{timestamp}_.h5"
    unique_model_path = os.path.join(trained_model_dir, unique_model_name)
    return unique_model_path
