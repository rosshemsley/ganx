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
        yield batch
