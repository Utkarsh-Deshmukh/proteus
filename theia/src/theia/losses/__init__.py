"""import all the available losses, and update the ledger."""

# pylint: disable=no-name-in-module
from theia.losses.loss_ledger import LOSS  # type: ignore
from theia.losses.two_way_multilabel import TwoWayLoss  # type: ignore
from theia.losses.dice_score import DiceScore  # type: ignore
