import argparse
import os
import logging
from src.utils import read_yaml, create_directories, create_and_save_tensorboard_callback, create_and_save_checkpoint_callback
import io



STAGE = "prepare_callback"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def prepare_callbacks(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    tensorboard_log_dir = os.path.join(artifacts_dir, artifacts["artifacts"]["TENSORBOARD_ROOT_LOG_DIR"])
    checkpoint_dir = os.path.join(artifacts_dir, artifacts["artifacts"]["CHECKPOINT_DIR"])
    callbacks_dir = os.path.join(artifacts_dir, artifacts["artifacts"]["CALLBACKS_DIR"])

    create_directories([
        tensorboard_log_dir,
        checkpoint_dir,
        callbacks_dir
    ])

    create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir)
    create_and_save_checkpoint_callback(callbacks_dir, checkpoint_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        prepare_callbacks(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e