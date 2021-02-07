from typing import Callable, Tuple

from omegaconf import DictConfig
import haiku as hk
import jax.numpy as jnp
import jax
import jax.nn as nn


class Critic(hk.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.conv_channels = cfg.model.conv_channels

    def __call__(self, img: jnp.ndarray) -> jnp.ndarray:
        from_rgb = _from_rgb(self.conv_channels)
        encoder = _encode(self.conv_channels)
        flatten = hk.Linear(1)

        x = from_rgb(img)
        x = encoder(x)
        return flatten(x.reshape(img.shape[0], -1))


class Generator(hk.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.conv_channels = cfg.model.conv_channels
        self.base_resolution = cfg.model.base_resolution

    def __call__(self, latent_vector: jnp.ndarray) -> jnp.ndarray:
        from_latent = _latent_decoder(self.base_resolution, self.conv_channels)
        decode = _decode(self.conv_channels)
        to_rgb = _to_rgb()

        x = from_latent(latent_vector)
        x = decode(x)
        x = to_rgb(x)

        return x


def random_latent_vectors(key: jnp.ndarray, n: int, cfg: DictConfig) -> jnp.ndarray:
    return jax.random.normal(key, shape=(n, cfg.model.latent_dims))


def _latent_decoder(
    base_resolution: Tuple[int, int], conv_channels: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return hk.Sequential(
        [
            hk.Linear(base_resolution[0] * base_resolution[1] * conv_channels),
            nn.leaky_relu,
            lambda x: x.reshape((-1, *base_resolution, conv_channels)),
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
            hk.GroupNorm(8),
            _downsample,
            _conv(channels),
            nn.leaky_relu,
            hk.GroupNorm(8),
            _downsample,
            _conv(channels),
            nn.leaky_relu,
            _downsample,
            _conv(channels),
            nn.leaky_relu,
        ]
    )


def _decode(channels: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return hk.Sequential(
        [
            hk.GroupNorm(8),
            _upsample,
            _conv(channels),
            nn.leaky_relu,
            hk.GroupNorm(8),
            _upsample,
            _conv(channels),
            nn.leaky_relu,
            hk.GroupNorm(8),
            _upsample,
            _conv(channels),
            nn.leaky_relu,
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
    return jax.image.resize(x, shape=(n, h // 2, w // 2, c), method="bilinear")


def _upsample(x: jnp.ndarray) -> jnp.ndarray:
    n, h, w, c = x.shape
    return jax.image.resize(x, shape=(n, h * 2, w * 2, c), method="bilinear")
