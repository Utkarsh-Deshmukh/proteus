"""Define the dice score. src= https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py"""

# pylint: disable=no-name-in-module
import torch
from torch import Tensor
from theia.losses import LOSS  # type: ignore


@LOSS.register_module
class DiceScore(torch.nn.Module):
    """Class defining the Dice score."""

    def __init__(self, multiclass: bool = False):
        """Create an object.

        Args:
            multiclass (bool, optional): is the classification task multiclass. Defaults to False.
        """
        self.multiclass = multiclass

    def forward(self, predictions: Tensor, ground_truth: Tensor) -> Tensor:
        """Run the forward routine for loss calculation.

        Args:
            predictions (Tensor): prediction values.
            ground_truth (Tensor): ground truth values.

        Returns:
            Tensor: computer loss.
        """
        # Dice loss (objective to minimize) between 0 and 1
        function = self.multiclass_dice_coeff if self.multiclass else self.dice_coeff
        return 1 - function(predictions, ground_truth, reduce_batch_first=True)

    def multiclass_dice_coeff(
        self, predictions: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6
    ) -> Tensor:
        """Computer multi class dice coefficients.

        Args:
            predictions (Tensor): prediction values.
            target (Tensor): ground truth values.
            reduce_batch_first (bool, optional): _description_. Defaults to False.
            epsilon (float, optional): _description_. Defaults to 1e-6.

        Returns:
            Tensor: computed loss
        """
        # Average of Dice coefficient for all classes
        return self.dice_coeff(predictions.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

    def dice_coeff(self, predictions: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
        """Compute dice coefficients.

        Args:
            predictions (Tensor): prediction values.
            target (Tensor): ground truth values.
            reduce_batch_first (bool, optional): _description_. Defaults to False.
            epsilon (float, optional): _description_. Defaults to 1e-6.

        Returns:
            Tensor: computed loss.
        """
        # Average of Dice coefficient for all batches, or for a single mask
        assert predictions.size() == target.size()
        assert predictions.dim() == 3 or not reduce_batch_first

        sum_dim = (-1, -2) if predictions.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (predictions * target).sum(dim=sum_dim)
        sets_sum = predictions.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()
