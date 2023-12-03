"""Test of DQ Floral Dataset Loader
"""

from typing import Final, List
from .dataset_loader import DQFloralDatasetLoader

IMG_PATHS: Final[List[str]] = [
    "/dq_floral_dataset/Species_0/Bloom_0/Beetle_0/image_50.png",
    "/dq_floral_dataset/Species_4/Bloom_1/Beetle_0/image_280.png",
    "/dq_floral_dataset/Species_0/Bloom_0/Beetle_1/image_173.png",
]
IMG_LABELS: Final[List[List[int]]] = [
    [0, 0, 0],
    [4, 1, 0],
][0, 0, 1]


class TestDQFloralDatasetLoader:
    ...
