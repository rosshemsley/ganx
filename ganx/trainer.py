from pathlib import Path
from omegaconf import DictConfig
import haiku as hk

from ganx.datasets import CelebADataset


def train(cfg: DictConfig, dataset_path: Path):
    dataset = CelebADataset(dataset_path)

    print("dataset has", len(dataset), "entries")
