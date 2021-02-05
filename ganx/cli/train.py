import pathlib

import click
import jax
from omegaconf import OmegaConf

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
    if debug:
        print("Disabling jit")
        jax.config.update("jax_disable_jit", True)

    cfg = OmegaConf.load(config_path)
    print(OmegaConf.to_yaml(cfg))

    train(cfg, pathlib.Path(root_dir))


if __name__ == "__main__":
    main()
