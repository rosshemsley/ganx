import pathlib
from datetime import datetime

import click
import jax
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from ganx.trainer import train


@click.command()
@click.option(
    "--root-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Root directory of the Celeb A dataset.",
)
@click.option(
    "--config-path",
    default="config.yaml",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Path to config",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    type=bool,
    help="Disable jit for debugging",
)
def main(config_path: str, root_dir: str, debug: bool) -> None:
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_readonly(cfg, True)

    print(OmegaConf.to_yaml(cfg))

    summary_writer = SummaryWriter()
    train(cfg, pathlib.Path(root_dir), summary_writer)


if __name__ == "__main__":
    main()
