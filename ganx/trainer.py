import jax
from pathlib import Path
from omegaconf import DictConfig
import haiku as hk

from ganx.datasets import CelebADataset, batch_iterator
from ganx.modules import Critic, Generator


def train(cfg: DictConfig, dataset_path: Path):
    key = jax.random.PRNGKey(cfg.random_seed)

    dataset = CelebADataset(dataset_path)
    critic = Critic(cfg)
    generator = Generator(cfg)

    for epoch in range(cfg.trainer.epochs):
        for batch_idx, batch in enumerate(batch_iterator(cfg.trainer.batch_size, dataset)):
            print(f"batch {batch_idx}")


