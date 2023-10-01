"""Training script.
"""
import logging
from typing import Final
from sklearn.model_selection import KFold
from data_loaders.dq_floral_dataset.dataset_collector import DQFloralDatasetCollector
from data_loaders.dq_floral_dataset.dataset_loader import DQFloralDatasetLoader
import utils.logging_functions as log_funcs

DATASET_PATH: Final[str] = "/dq_floral_dataset"
NUM_OF_CROSS_VAL_SPLITS: Final[int] = 7
TRAINING_BATCH_SIZE: Final[int] = 2

def main():
    logging.root.setLevel(logging.INFO)
    
    # dataset collection and splits setup
    dataset_collector = DQFloralDatasetCollector(DATASET_PATH)
    collected_dataset = dataset_collector.get_dataset()
    kfold = KFold(n_splits=NUM_OF_CROSS_VAL_SPLITS, shuffle=True, random_state=dataset_collector.seed)
    
    for split_ind, (train_split_indices, val_split_indices) in enumerate(kfold.split(collected_dataset["img_paths"])):
        logging.info("/"*50 + f" K-fold split {split_ind + 1} of {NUM_OF_CROSS_VAL_SPLITS} " + "/"*50)
        
        # get training and validation dataset
        train_data_img_paths = [collected_dataset["img_paths"][ind] for ind in train_split_indices]
        train_data_img_labels = [collected_dataset["labels"][ind] for ind in train_split_indices]
        val_data_img_paths = [collected_dataset["img_paths"][ind] for ind in val_split_indices]
        val_data_img_labels = [collected_dataset["labels"][ind] for ind in val_split_indices]
        
        log_funcs.log_kfold_split_info(len(train_data_img_paths), len(val_data_img_paths))
        
        # create TF Dataset
        train_dataset_loader = DQFloralDatasetLoader(dataset_collector, train_data_img_paths, train_data_img_labels, shuffle=True, batch_size=TRAINING_BATCH_SIZE)
        train_data = train_dataset_loader()
        val_dataset_loader = DQFloralDatasetLoader(dataset_collector, val_data_img_paths, val_data_img_labels, shuffle=False, batch_size=TRAINING_BATCH_SIZE)
        val_data = val_dataset_loader()
        
        # build the model
        

        


main()