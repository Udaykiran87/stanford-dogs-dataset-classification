import argparse
import os
import logging
import matplotlib.pyplot as plt
import pickle
from src.utils import read_yaml, create_directories, load_full_model, get_latest_pretrained_model, train_valid_generator

STAGE = "evaluate"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def evaluate_model(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    model_eval_metrics_dir_path = os.path.join(artifacts_dir, artifacts["MODEL_EVAL_METRICS_DIR"])
    create_directories([model_eval_metrics_dir_path])

    model_metrics_dir_path = os.path.join(artifacts_dir, artifacts["MODEL_TRAIN_METRICS_DIR"])

    trained_model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    latest_model_file = get_latest_pretrained_model(trained_model_dir)

    model = load_full_model(latest_model_file)

    _, valid_generator = train_valid_generator(
        config,
        params
    )
    batch_size = params["model"]["BATCH_SIZE"]

    (eval_loss, eval_accuracy) = model.evaluate(valid_generator, batch_size= batch_size, verbose= 1)
    logging.info(f"Validation Loss: {eval_loss}")
    logging.info(f"Validation Accuracy: {eval_accuracy}")

    model_history_file = os.path.join(model_metrics_dir_path, "trainHistoryDict")
    with open(model_history_file, "rb") as model_history:
        history = pickle.load(model_history)

    acc_figure_file = os.path.join(model_eval_metrics_dir_path, 'baseline_acc_epoch.png')
    plt.subplot()
    plt.title('Model Accuracy')
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training Accuracy','Validation Accuracy'])
    plt.savefig(acc_figure_file, transparent= False, bbox_inches= 'tight', dpi= 900)

    loss_figure_file = os.path.join(model_eval_metrics_dir_path, 'baseline_loss_epoch.png')
    plt.title('Model Loss')
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training Loss','Validation Loss'])
    plt.savefig(loss_figure_file, transparent= False, bbox_inches= 'tight', dpi= 900)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        evaluate_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
