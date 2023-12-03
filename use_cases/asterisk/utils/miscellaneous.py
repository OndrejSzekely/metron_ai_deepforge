from os import path
import os
from shutil import rmtree
import numpy as np
import tensorflow as tf


def create_tb_logging_folders(tb_logging_folder, exp_tb_logging_folder):
    if not path.exists(tb_logging_folder):
        os.mkdir(tb_logging_folder)
    if not path.exists(exp_tb_logging_folder):
        os.mkdir(exp_tb_logging_folder)
    else:
        rmtree(exp_tb_logging_folder)
        os.mkdir(exp_tb_logging_folder)
        
def compute_class_image_num(labels):
    labels_np = np.array(labels)
    res = {"species": [0]*6, "bloom": [0]*2, "beetle": [0]*2}
    for ind in range(6):
        res["species"][ind] = np.sum(np.where(labels_np[:, 0] == ind, 1, 0))
    for ind in range(2):
        res["bloom"][ind] = np.sum(np.where(labels_np[:, 1] == ind, 1, 0))
    for ind in range(2):
        res["beetle"][ind] = np.sum(np.where(labels_np[:, 2] == ind, 1, 0))
    return res


def tf_log_labels_stat(prefix, train_fold_labels_stat, kfold_step):
    full_prefix = f"kfold_labels_{prefix}"
    tf.summary.scalar(f"{full_prefix}/species_0", train_fold_labels_stat["species"][0], step=kfold_step)
    tf.summary.scalar(f"{full_prefix}/species_1", train_fold_labels_stat["species"][1], step=kfold_step)
    tf.summary.scalar(f"{full_prefix}/species_2", train_fold_labels_stat["species"][2], step=kfold_step)
    tf.summary.scalar(f"{full_prefix}/species_3", train_fold_labels_stat["species"][3], step=kfold_step)
    tf.summary.scalar(f"{full_prefix}/species_4", train_fold_labels_stat["species"][4], step=kfold_step)
    tf.summary.scalar(f"{full_prefix}/species_5", train_fold_labels_stat["species"][5], step=kfold_step)
    
    tf.summary.scalar(f"{full_prefix}/bloom_0", train_fold_labels_stat["bloom"][0], step=kfold_step)
    tf.summary.scalar(f"{full_prefix}/bloom_1", train_fold_labels_stat["bloom"][1], step=kfold_step)
    
    tf.summary.scalar(f"{full_prefix}/beetle_0", train_fold_labels_stat["beetle"][0], step=kfold_step)
    tf.summary.scalar(f"{full_prefix}/beetle_1", train_fold_labels_stat["beetle"][1], step=kfold_step)

def tf_log_history(prefix, history, step):
    for key, value in history.items():
        tf.summary.scalar(f"metrics_{prefix}/{key}", value[-1], step=step)
        
def tf_log_full_history(prefix, history):
    for key, value in history.items():
        for ind in range(len(value)):
            tf.summary.scalar(f"metrics_{prefix}/{key}", value[ind], step=ind)