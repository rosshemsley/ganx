import pathlib
from PIL import Image
from chex import dataclass
from typing import Sequence


@dataclass
class CelebASample:
    img: Image


def load_celeb_a(path: pathlib.Path) -> Sequence[CelebASample]:
    return []
