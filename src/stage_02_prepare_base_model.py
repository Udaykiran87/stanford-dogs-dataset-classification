import argparse
import os
import io
import shutil
from tqdm import tqdm
import logging
import argparse
import pandas as pd
from src.utils import read_yaml, create_directories, get_inceptionV3_model, prepare_model


STAGE = "prepare_base_model" 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    base_model_dir = artifacts["BASE_MODEL_DIR"]
    base_model_name = artifacts["BASE_MODEL_NAME"]

    base_model_dir_path = os.path.join(artifacts_dir, base_model_dir)
    create_directories([base_model_dir_path])

    base_model_path = os.path.join(base_model_dir_path, base_model_name)

    input_shape = (params["preprocess"]["img_width"], params["preprocess"]["img_height"], params["preprocess"]["channels"])
    model = get_inceptionV3_model(
        input_shape = input_shape,
        model_path = base_model_path
        )

    full_model = prepare_model(
        model = model, 
        CLASSES = params["model"]["CLASSES"], 
        freeze_all = True, 
        freeze_till = None, 
        learning_rate = params["model"]["LEARNING_RATE"]
        )

    update_base_model_path = os.path.join(
        base_model_dir_path,
        artifacts["UPDATED_BASE_MODEL_NAME"]
    )

    def _log_model_summary(full_model):
        with io.StringIO() as stream:
            full_model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    logging.info(f"full model summary: \n{_log_model_summary(full_model)}")

    full_model.save(update_base_model_path)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e