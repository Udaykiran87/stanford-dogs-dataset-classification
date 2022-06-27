import tensorflow as tf
import os
import joblib
import logging
from src.utils import get_timestamp

def create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir):
    unique_name = get_timestamp("tb_logs")

    tb_running_log_dir = os.path.join(tensorboard_log_dir, unique_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    tb_callback_filepath = os.path.join(callbacks_dir, "tensorboard_cb.cb")
    joblib.dump(tensorboard_callback, tb_callback_filepath)
    logging.info(f"tensorboard callback is being saved at {tb_callback_filepath}")


def create_and_save_checkpoint_callback(callbacks_dir, checkpoint_dir):
    checkpoint_file_path = os.path.join(checkpoint_dir, "ckpt_model.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_file_path,
        save_best_only = True,
        monitor='val_loss',
        verbose=1,
        mode='auto',
        save_weights_only=False,
        period=1
    )

    ckpt_callback_filepath = os.path.join(callbacks_dir, "checkpoint_cb.cb")
    joblib.dump(checkpoint_callback, ckpt_callback_filepath)
    logging.info(f"checkpoint callback is being saved at {checkpoint_callback}")

def create_and_save_earlystop_callback(callbacks_dir):
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=3,
        verbose=1,
        mode='auto'
    )

    earlystop_callback_filepath = os.path.join(callbacks_dir, "earlystop_cb.cb")
    joblib.dump(earlystop_callback, earlystop_callback_filepath)
    logging.info(f"earlystop callback is being saved at {earlystop_callback}")

def create_and_save_csvlogger_callback(callbacks_dir, csvlogger_dir):
    csvlogger_file_path = os.path.join(csvlogger_dir, "training_csv.log")
    csvlogger_callback = tf.keras.callbacks.CSVLogger(
        filename= csvlogger_file_path,
        separator = ",",
        append = False
    )

    csvlogger_callback_filepath = os.path.join(callbacks_dir, "csvlogger_cb.cb")
    joblib.dump(csvlogger_callback, csvlogger_callback_filepath)
    logging.info(f"csvlogger callback is being saved at {csvlogger_callback}")

def create_and_save_reduceLR_callback(callbacks_dir):
    reduceLR_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1,
        mode='auto'
    )

    reduceLR_callback_filepath = os.path.join(callbacks_dir, "reduceLR_cb.cb")
    joblib.dump(reduceLR_callback, reduceLR_callback_filepath)
    logging.info(f"reduceLR callback is being saved at {reduceLR_callback}")

def get_callbacks(callback_dir_path):
    callback_path = [
        os.path.join(callback_dir_path, bin_file) for bin_file in os.listdir(callback_dir_path) if bin_file.endswith(".cb")
    ]

    callbacks = [
        joblib.load(path) for path in callback_path
    ]

    logging.info(f"saved callbacks are loaded from {callback_dir_path}")

    return callbacks
