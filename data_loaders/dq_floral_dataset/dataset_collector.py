"""
DQ Floral Dataset Collector
"""


from typing import Dict, List, Final
import os
import logging
from os import path
from metron_shared import param_validators as param_val

class DQFloralDatasetCollector:
    """DQ Floral Dataset collector
    
    Collects all image paths and corresponding labels.
    
    Data collection type: In-memory
    RAM requirements: N/A
    
    Atributes:
        root_folder (str): Dataset file system root path.
        seed (int): Random seed.
        _species (Final[List[str]]): List of flower classes. Class id is given by the order in the list.
        _bloom (Final[List[str]]): List of bloom type classes. Class id is given by the order in the list.
        _beetle (Final[List[str]]): List of beetle type classes. Class id is given by the order in the list.
        _img_paths (List[str]]): List of collected images.
        _img_labeles (List[List[int]]]): List of images labels.
        _stats (List[Tuple[str, int]]): Dictionary with datasets statistics.
        species (Final[str]): Recognition task name.
        bloom (Final[str]): Recognition task name.
        beetle (Final[str]): Recognition task name.
    """
    
    _species: Final[List[str]] = ["Species_0", "Species_1", "Species_2", "Species_3" ,"Species_4", "Species_5"]
    _bloom: Final[List[str]] = ["Bloom_0", "Bloom_1"]
    _beetle: Final[List[str]] = ["Beetle_0", "Beetle_1"]
    species: Final[str] = "species"
    bloom: Final[str] = "bloom"
    beetle: Final[str] = "beetle"
    
    
    def __init__(self, dataset_root_folder_path: str, seed: int = 1931992) -> None:
        
        param_val.check_type(dataset_root_folder_path, str)
        param_val.check_type(seed, int)

        self.root_folder = dataset_root_folder_path
        self.seed = seed
        self._img_paths = []
        self._img_labeles = []
        self._stats = []
        self._walk_through_folders()
        self._print_statistics()
    
    def get_number_of_species_classes(self) -> int:
        return len(self._species)
    
    def get_number_of_bloom_classes(self) -> int:
        return len(self._bloom)
    
    def get_number_of_beetle_classes(self) -> int:
        return len(self._beetle)
    
    def _get_image_label_encoding(self, species_label: str, bloom_label: str, beetle_label: str) -> List[int]:
        """Converts class names into labels index vector of the image.

        Args:
            species_label (str): Species class name from <self._species>
            bloom_label (str): Bloom class name from <self._bloom>
            beetle_label (str): Beetle class name from <self._beetle>

        Returns:
            List[int]: List of 3 items. Class label id for given class types. The order is following: species, bloom, beetle.
        """
        
        param_val.check_type(species_label, str)
        param_val.check_type(bloom_label, str)
        param_val.check_type(beetle_label, str)
        
        return [self._species.index(species_label), self._bloom.index(bloom_label), self._beetle.index(beetle_label)]
    
    def _walk_through_folders(self) -> None:
        """Walks through dataset's subfolders and gathers image paths.
        """
        for species_label in self._species:
            for bloom_label in self._bloom:
                for beetle_label in self._beetle:
                    images_gt_path = path.join(self.root_folder, species_label, bloom_label, beetle_label)
                    
                    images_num = 0
                    if path.exists(images_gt_path):
                        for img_file in os.scandir(images_gt_path):
                            self._img_paths.append(img_file.path)
                            self._img_labeles.append(self._get_image_label_encoding(species_label, bloom_label, beetle_label))
                            images_num +=1
                    
                    self._stats.append((f"{species_label}/{bloom_label}/{beetle_label}:", images_num))

    def get_dataset(self) -> Dict[str, List]:
        """Returns the dataset.

        Returns:
            Dict[str, List]: Collected dataset.
        """
        return {
            "img_paths": self._img_paths,
            "labels": self._img_labeles
        }

    def _print_statistics(self) -> None:
        logging.info("*"*15 + " DQ Floral Dataset Collector Stats " + "*"*15)
        for stat_record in self._stats:
            logging.info(f"{stat_record[0]} {stat_record[1]}")
        logging.info("*"*65)