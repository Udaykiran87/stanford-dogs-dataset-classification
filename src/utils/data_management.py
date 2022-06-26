import tensorflow as tf
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_valid_generator(config, params):
    if params["preprocess"]["do_data_augmentation"]:
        shear_range = params["preprocess"]["shear_range"]
        zoom_range = params["preprocess"]["zoom_range"]
        horizontal_flip = params["preprocess"]["horizontal_flip"]
        rotation_range = params["preprocess"]["rotation_range"]
        width_shift_range = params["preprocess"]["width_shift_range"]
        height_shift_range = params["preprocess"]["height_shift_range"]
        validation_split = params["preprocess"]["validation_split"]

        train_datagen = ImageDataGenerator(
            rescale= 1./255,
            shear_range= shear_range,
            zoom_range= zoom_range,
            horizontal_flip= horizontal_flip,
            rotation_range= rotation_range,
            width_shift_range= width_shift_range,
            height_shift_range= height_shift_range,
            validation_split=validation_split,
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale= 1./255,
            validation_split=validation_split,
        )

    validation_split = params["preprocess"]["validation_split"]
    valid_datagen = ImageDataGenerator(
        rescale= 1./255,
        validation_split=0.2,
    )

    train_data_dir = config["artifacts"]["TRAIN_DATA_DIR"]
    img_width = params["preprocess"]["img_width"]
    img_height = params["preprocess"]["img_height"]
    color_mode = params["preprocess"]["color_mode"]
    batch_size = params["preprocess"]["batch_size"]
    class_mode = params["preprocess"]["class_mode"]
    shuffle = params["preprocess"]["shuffle"]
    seed = params["preprocess"]["seed"]

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size= (img_width, img_height),
        color_mode= color_mode,
        batch_size= batch_size,
        class_mode= class_mode,
        subset='training',
        shuffle= shuffle,
        seed= seed
    )

    valid_generator = valid_datagen.flow_from_directory(
        train_data_dir,
        target_size= (img_width, img_height),
        color_mode= color_mode,
        batch_size= batch_size,
        class_mode= class_mode,
        subset='validation',
        shuffle= shuffle,
        seed= seed
    )

    logging.info("train and valid generator is created.")
    return train_generator, valid_generator
