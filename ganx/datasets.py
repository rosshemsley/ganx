import jax.numpy as jnp

from typing import Sequence, Any

import pathlib
from PIL import Image


class CelebADataset:
    def __init__(self, root_dir: pathlib.Path):
        self.paths = [p for p in root_dir.iterdir() if str(p).endswith(".jpg")]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Image:
        return Image.open(self.paths[idx])


def batch_iterator(batch_size: int, dataset: Sequence[Any]) -> Sequence[Any]:
    n = len(dataset)
    i = 0

    while i < n:
        batch = []

        for k in range(i, min(n, i + batch_size)):
            batch.append(dataset[k])

        i += len(batch)
        if len(batch) == batch_size:
            yield _to_tensor(batch)


def _to_tensor(batch: Sequence[Image.Image]) -> jnp.ndarray:
    x = (
        1 / 128 * jnp.stack([jnp.asarray(img) for img in batch]).astype(jnp.float32)
    ) - 1.0
    return x
