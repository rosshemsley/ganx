import pathlib
from PIL import Image


class CelebADataset:
    def __init__(self, root_dir: pathlib.Path):
        self.paths = [p for p in root_dir.iterdir() if str(p).endswith(".jpg")]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Image:
        return Image(self.paths[idx])
