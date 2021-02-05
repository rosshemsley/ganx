from typing import Any, Iterable, Sequence, Tuple


import optax
import jax
import jax.numpy as jnp
from pathlib import Path
from omegaconf import DictConfig
import haiku as hk
from PIL import Image

from ganx.datasets import CelebADataset, batch_iterator
from ganx.modules import Critic, Generator, random_latent_vectors


OptState = Any
ImgBatch = Sequence[jnp.ndarray]
LatentBatch = Sequence[jnp.ndarray]
RNG = jnp.ndarray


def train(cfg: DictConfig, dataset_path: Path) -> None:
    rng = jax.random.PRNGKey(cfg.random_seed)
    dataset = CelebADataset(dataset_path)

    generator_opt = optax.adam(cfg.opt.lr)
    critic_opt = optax.adam(cfg.opt.lr)

    def generator_fn(x: jnp.ndarray) -> jnp.ndarray:
        return Generator(cfg)(x)

    def critic_fn(x: jnp.ndarray) -> jnp.ndarray:
        return Critic(cfg)(x)

    generator = hk.without_apply_rng(hk.transform(generator_fn))
    critic = hk.without_apply_rng(hk.transform(critic_fn))

    @jax.jit
    def generator_loss(
        generator_params: hk.Params, latent_batch: LatentBatch
    ) -> jnp.ndarray:

        f_l = generator.apply(generator_params, latent_batch)

        return -jnp.mean(f_l)

    @jax.jit
    def critic_loss(
        critic_params: hk.Params,
        generator_params: hk.Params,
        img_batch: ImgBatch,
        latent_batch: LatentBatch,
    ) -> jnp.ndarray:

        f_l = critic.apply(
            critic_params, generator.apply(generator_params, latent_batch)
        )
        f_x = critic.apply(critic_params, img_batch)

        # TODO(Ross): gradient penalty... signs.

        return jnp.mean(f_l) - jnp.mean(f_x)

    @jax.jit
    def update_critic(
        img_batch: ImgBatch,
        latent_batch: LatentBatch,
        critic_params: hk.Params,
        generator_params: hk.Params,
        opt_state: OptState,
    ) -> Tuple[hk.Params, OptState]:

        loss, grad = jax.value_and_grad(critic_loss)(
            critic_params, generator_params, img_batch, latent_batch
        )
        updates, opt_state = critic_opt.update(grad, opt_state)
        new_params = optax.apply_updates(critic_params, updates)

        return loss, new_params, opt_state

    @jax.jit
    def update_generator(
        latent_batch: LatentBatch,
        params: hk.Params,
        opt_state: OptState,
    ) -> Tuple[hk.Params, OptState]:

        loss, grad = jax.value_and_grad(generator_loss)(params, latent_batch)
        updates, opt_state = generator_opt.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)

        return loss, new_params, opt_state

    generator_params = generator.init(rng, _latent_batch(rng, cfg))
    critic_params = critic.init(rng, _dummy_image(cfg))

    generator_opt_state = generator_opt.init(generator_params)
    critic_opt_state = critic_opt.init(critic_params)

    for epoch in range(cfg.trainer.epochs):
        for batch_idx, img_batch in enumerate(_batch_iter(cfg, dataset)):
            print(f"batch {batch_idx}")

            latent = _latent_batch(rng, cfg)
            loss, critic_params, critic_opt_state = update_critic(
                img_batch,
                latent,
                critic_params,
                generator_params,
                critic_opt_state,
            )

            print("loss", loss)

            if batch_idx % cfg.trainer.generator_step == 0:
                loss, generator_params, generator_opt_state = update_generator(
                    latent, generator_params, generator_opt_state
                )


def _batch_iter(
    cfg: DictConfig, dataset: Sequence[jnp.ndarray]
) -> Iterable[jnp.ndarray]:
    res = _output_resolution(cfg)
    for b in batch_iterator(cfg.trainer.batch_size, dataset):
        n, _, __, c = b.shape
        yield jax.image.resize(b, shape=(n, *res, c), method="bilinear")


def _dummy_image(cfg: DictConfig) -> ImgBatch:
    res = _output_resolution(cfg)
    return jnp.zeros((cfg.trainer.batch_size, *res, 3))


def _output_resolution(cfg: DictConfig) -> Tuple[int, int]:
    h, w = cfg.model.base_resolution
    return h * 2 ** 2, w * 2 ** 2


def _latent_batch(rng: RNG, cfg: DictConfig):
    return random_latent_vectors(rng, cfg.trainer.batch_size, cfg)
