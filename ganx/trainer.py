from typing import Any, Dict, Iterable, Sequence, Tuple

from tensorboardX import SummaryWriter
from chex import assert_shape
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
Log = Dict[str, Any]


def train(cfg: DictConfig, dataset_path: Path, writer: SummaryWriter) -> None:
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
        generator_params: hk.Params, critic_params: hk.Params, latent_batch: LatentBatch
    ) -> jnp.ndarray:
        batch_size = latent_batch.shape[0]

        img_gen = generator.apply(generator_params, latent_batch)
        f_gen = critic.apply(critic_params, img_gen)

        assert_shape(f_gen, (batch_size, 1))

        return -jnp.mean(f_gen)

    @jax.jit
    def critic_loss(
        critic_params: hk.Params,
        generator_params: hk.Params,
        img_batch: ImgBatch,
        latent_batch: LatentBatch,
    ) -> Tuple[jnp.ndarray, Log]:

        batch_size = img_batch.shape[0]
        img_generated = generator.apply(generator_params, latent_batch)

        def f(x):
            return critic.apply(critic_params, x)

        f_real = f(img_batch)

        # Use the vector-jacobian product to efficiently compute the grad for each f_i.
        f_gen, grad_fn = jax.vjp(f, img_generated)
        grad = grad_fn(jnp.ones(f_gen.shape))[0]

        assert_shape(f_real, (batch_size, 1))
        assert_shape(f_gen, (batch_size, 1))
        assert_shape(grad, img_batch.shape)

        flat_grad = grad.reshape(batch_size, -1)
        gp = jnp.square(1 - jnp.linalg.norm(flat_grad, axis=1))
        assert_shape(gp, (batch_size,))

        loss = jnp.mean(f_gen) - jnp.mean(f_real)
        gradient_penalty = jnp.mean(gp)

        log = {
            "wasserstein": -loss,
            "gradient_penalty": gradient_penalty,
        }

        return loss + 10 * gradient_penalty, log

    @jax.jit
    def update_critic(
        img_batch: ImgBatch,
        latent_batch: LatentBatch,
        critic_params: hk.Params,
        generator_params: hk.Params,
        opt_state: OptState,
    ) -> Tuple[hk.Params, OptState, Log]:

        (loss, log), grad, = jax.value_and_grad(
            critic_loss, has_aux=True
        )(critic_params, generator_params, img_batch, latent_batch)
        updates, opt_state = critic_opt.update(grad, opt_state)
        new_params = optax.apply_updates(critic_params, updates)

        return loss, new_params, opt_state, log

    @jax.jit
    def update_generator(
        latent_batch: LatentBatch,
        generator_params: hk.Params,
        critic_params: hk.Params,
        opt_state: OptState,
    ) -> Tuple[hk.Params, OptState]:

        loss, grad = jax.value_and_grad(generator_loss)(
            generator_params, critic_params, latent_batch
        )
        updates, opt_state = generator_opt.update(grad, opt_state)
        new_params = optax.apply_updates(generator_params, updates)

        return loss, new_params, opt_state

    rng, key = jax.random.split(rng)
    generator_params = generator.init(
        key,
        _dummy_latent(
            cfg,
        ),
    )
    rng, key = jax.random.split(rng)
    critic_params = critic.init(key, _dummy_image(cfg))

    generator_opt_state = generator_opt.init(generator_params)
    critic_opt_state = critic_opt.init(critic_params)

    for epoch in range(cfg.trainer.epochs):
        for batch_idx, total_batches, img_batch in _batch_iter(cfg, dataset):
            rng, latent = _latent_batch(rng, cfg)
            loss, critic_params, critic_opt_state, log = update_critic(
                img_batch,
                latent,
                critic_params,
                generator_params,
                critic_opt_state,
            )

            writer.add_scalar('loss/wasserstein', log["wasserstein"])
            writer.add_scalar('loss/gradient_penalty', log["gradient_penalty"])
            writer.add_scalar('loss/loss', loss)

            if batch_idx % 10 == 0:
                print(f"({batch_idx}/{total_batches}) loss: {loss}, {log}")

            if batch_idx % cfg.trainer.generator_step == 0:
                rng, latent = _latent_batch(rng, cfg)
                loss, generator_params, generator_opt_state = update_generator(
                    latent, generator_params, critic_params, generator_opt_state
                )


def _batch_iter(
    cfg: DictConfig, dataset: Sequence[jnp.ndarray]
) -> Iterable[jnp.ndarray]:
    res = _output_resolution(cfg)
    n_batches = len(dataset) // cfg.trainer.batch_size

    for i, b in enumerate(batch_iterator(cfg.trainer.batch_size, dataset)):
        n, _, __, c = b.shape
        yield i, n_batches, jax.image.resize(b, shape=(n, *res, c), method="bilinear")


def _dummy_image(cfg: DictConfig) -> ImgBatch:
    res = _output_resolution(cfg)
    return jnp.zeros((cfg.trainer.batch_size, *res, 3))


def _dummy_latent(cfg: DictConfig) -> LatentBatch:
    return jnp.zeros((cfg.trainer.batch_size, cfg.model.latent_dims))


def _output_resolution(cfg: DictConfig) -> Tuple[int, int]:
    h, w = cfg.model.base_resolution
    return h * 2 ** 2, w * 2 ** 2


def _latent_batch(rng: RNG, cfg: DictConfig) -> Tuple[RNG, LatentBatch]:
    rng, key = jax.random.split(rng)
    return rng, random_latent_vectors(key, cfg.trainer.batch_size, cfg)
