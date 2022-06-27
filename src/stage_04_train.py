import argparse
import os
import logging
from src.utils import read_yaml, create_directories, load_full_model, get_callbacks, train_valid_generator, \
    get_unique_path_to_save_model
from tensorflow.keras.utils import to_categorical

STAGE = "train"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def train_model(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    train_model_dir_path = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directories([train_model_dir_path])

    untrained_full_model_path = os.path.join(artifacts_dir, artifacts["BASE_MODEL_DIR"], artifacts["UPDATED_BASE_MODEL_NAME"])

    model = load_full_model(untrained_full_model_path)

    callback_dir_path = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callback_dir_path)

    train_generator, valid_generator = train_valid_generator(
        config,
        params
    )

    num_classes = len(train_generator.class_indices)
    train_labels = train_generator.classes
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    valid_labels = valid_generator.classes
    valid_labels = to_categorical(valid_labels, num_classes=num_classes)

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    history = model.fit(
    train_generator,
    epochs = params["EPOCHS"],
    steps_per_epoch = steps_per_epoch,
    validation_data = valid_generator,
    validation_steps = validation_steps,
    verbose = 2,
    callbacks = callbacks,
    shuffle = True
    )

    logging.info(f"training completed")

    trained_model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directories([trained_model_dir])
    model_file_path = get_unique_path_to_save_model(trained_model_dir)
    model.save(model_file_path)
    logging.info(f"trained model is saved at: {model_file_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e