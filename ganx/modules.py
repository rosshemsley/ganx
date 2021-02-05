from typing import Callable

from omegaconf import DictConfig
import haiku as hk
import jax.numpy as jnp
import jax
import jax.nn as nn


class Critic(hk.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.latent_dims = cfg.model.latent_dims
        self.conv_channels = cfg.model.conv_channels

    def __call__(self, img: jnp.ndarray) -> jnp.ndarray:
        from_rgb = _from_rgb(self.conv_channels)
        encoder = _encode(self.conv_channels)
        flatten = hk.Linear(1)

        return flatten(encoder(from_rgb(img)))


class Generator(hk.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.latent_dims = cfg.model.latent_dims

    def __call__(self, latent_vector: jnp.ndarray) -> jnp.ndarray:
        decoder = _latent_decoder(self.latent_dims)
        feature_extractor = _decode(self.conv_channels)
        to_rgb = _to_rgb()

        return to_rgb(_decode(decoder(latent_vector)))


def random_latent_vectors(key: jnp.ndarray, n: int, cfg: DictConfig) -> jnp.ndarray:
    return jax.random.normal(key, shape=(n, cfg.model.latent_dims))


def _latent_decoder(latent_dims: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return hk.Sequential(
        [
            hk.Linear(latent_dims),
            nn.relu,
        ]
    )


def _to_rgb() -> Callable[[jnp.ndarray], jnp.ndarray]:
    return hk.Sequential(
        [
            hk.Conv2D(
                output_channels=3,
                kernel_shape=1,
                stride=1,
                padding="SAME",
            ),
            jnp.tanh,
        ]
    )


def _from_rgb(channels: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return hk.Sequential(
        [
            hk.Conv2D(
                output_channels=channels,
                kernel_shape=1,
                stride=1,
                padding="SAME",
            ),
        ]
    )


def _encode(channels: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return hk.Sequential(
        [
            _downsample,
            _conv(channels),
            nn.relu,
            _downsample,
            _conv(channels),
            nn.relu,
        ]
    )


def _decode(channels: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return hk.Sequential(
        [
            _upsample,
            _conv(channels),
            nn.relu,
            _upsample,
            _conv(channels),
            nn.relu,
        ]
    )


def _conv(out_channels: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return hk.Conv2D(
        output_channels=out_channels,
        kernel_shape=3,
        stride=1,
        padding="SAME",
    )


def _downsample(x: jnp.ndarray) -> jnp.ndarray:
    """
    Half the resolution of the input NHWC tensor
    """
    n, h, w, c = x.shape
    return jax.image.resize(x, shape=(n, h // 2, w // 2), method="bilinear")


def _upsample(x: jnp.ndarray) -> jnp.ndarray:
    n, h, w, c = x.shape
    return jax.image.resize(x, shape=(n, h * 2, w * 2), method="bilinear")
