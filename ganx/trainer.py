from omegaconf import DictConfig
import haiku as hk


def train(cfg: DictConfig):
    hk.transform(generator)
    hk.transform(descriminator)
