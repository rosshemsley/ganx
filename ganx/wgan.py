import hk as haiku
import jax.numpy as jnp


def random_latent_vector(key):
    ...


class WGAN(hk.Module):
    def __call__(self, latent_vector: jnp.ndarray) -> jnp.ndarray:
        ...

    def discriminator(self, img: jnp.ndarray) -> jnp.ndarray:
        ...

    def generator(self, latent_vector: jnp.ndarray) -> jnp.ndarray:
        ...
