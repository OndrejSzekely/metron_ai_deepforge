"""Testing of DQ Floral Dataset Collector
"""

from typing import Final
from metron_shared import param_validators as param_val
from .dataset_collector import DQFloralDatasetCollector

DATASET_PATH: Final[str] = "/dq_floral_dataset"

class TestDQFloralDatasetCollector:
    
    def test_dataset_loading(self):
        """Test if dataset is loaded without error.
        """
        DQFloralDatasetCollector(DATASET_PATH)

    def test_label_encoding(self):
        """Test label encoding.
        """
        dataset_collector = DQFloralDatasetCollector(DATASET_PATH)
        label_encoding = dataset_collector._get_image_label_encoding("Species_1", "Bloom_0", "Beetle_1")
        
        assert len(label_encoding) == 3
        assert label_encoding[0] == 1 and label_encoding[1] == 0 and label_encoding[2] == 1

    def test_dataset_getter(self):
        """Test return of the dataset.
        """
        dataset_collector = DQFloralDatasetCollector(DATASET_PATH)
        dataset = dataset_collector.get_dataset()
        
        dataset_collector_aux = DQFloralDatasetCollector(DATASET_PATH)
        dataset_aux = dataset_collector_aux.get_dataset()
        
        assert "img_paths" in dataset
        assert "labels" in dataset
        assert len(dataset["img_paths"]) > 0
        assert len(dataset["img_paths"]) == len(dataset["labels"])
        assert (dataset["img_paths"][3] == dataset_aux["img_paths"][3]) and (dataset["img_paths"][15] == dataset_aux["img_paths"][15])