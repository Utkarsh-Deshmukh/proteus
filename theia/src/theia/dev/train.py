"""Module to facilitate training of network."""

# pylint: disable=no-name-in-module
# pylint: disable=W0511

from config import config
from theia.losses import LOSS  # type: ignore


if __name__ == "__main__":
    loss = LOSS.compile(config.get("loss", None))
