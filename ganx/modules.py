import haiku as hk
import jax.numpy as jnp
import jax
import jax.nn as nn

from omegaconf import DictConfig


class Descriminator(hk.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.latent_dims = cfg.model.latent_dims
        self.conv_channels = cfg.model.conv_channels

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        decoder = _latent_decoder(self.latent_dims)
        feature_extractor = _feature_extractor(self.conv_channels)
        to_rgb = _to_rgb()

        return to_rgb(feature_mapper(decoder(x)))


class Generator(hk.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.latent_dims = cfg.model.latent_dims

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ...


def _latent_decoder(latent_dims: int) -> Callable[jnp.ndarray, jnp.ndarray]:
    return nn.Sequential(
        [
            hk.Linear(self.latent_dims),
            nn.relu,
        ]
    )


def _to_rgb() -> Callable[jnp.ndarray, jnp.ndarray]:
    return nn.Sequential(
        [
            hk.Conv2D(
                output_channels=3,
                kernel_shape=1,
                stride=1,
                padding=0,
            ),
            nn.tanh,
        ]
    )


def _feature_extractor(channels: int) -> Callable[jnp.ndarray, jnp.ndarray]:
    return nn.Sequential(
        [
            _conv(channels),
            nn.relu,
            _conv(channels),
            nn.relu,
        ]
    )


def _conv(out_channels: int) -> Callable[jnp.ndarray, jnp.ndarray]:
    return hk.Conv2D(
        output_channels=out_channels,
        kernel_shape=3,
        stride=1,
        padding=1,
    )
