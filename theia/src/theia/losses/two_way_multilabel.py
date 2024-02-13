"""Define the Two way multi label loss.
src: https://github.com/tk1980/TwoWayMultiLabelLoss/blob/main/utils/criterion.py"""
# pylint: disable=no-name-in-module
import torch
from torch import nn
from torch import Tensor
from theia.losses import LOSS  # type: ignore

NEG_INFINITE = -100


@LOSS.register_module
class TwoWayLoss(nn.Module):
    """Define the Two way loss functions."""

    def __init__(self, temperature_pos=4.0, temperature_neg=1.0):
        """Instantiate the object.

        Args:
            temperature_pos (float, optional): temperature associated with positive samples. Defaults to 4.0.
            temperature_neg (float, optional): temperature associated with negative samples. Defaults to 1.0.
        """
        super().__init__()
        self.temperature_pos = temperature_pos
        self.temperature_neg = temperature_neg

    def forward(self, predictions: Tensor, ground_truth: Tensor) -> Tensor:
        """Run the forward pass for loss computation.

        Args:
            predictions (Tensor): predictions.
            ground_truth (Tensor): ground truth values.

        Returns:
            Tensor: computed loss
        """
        class_mask = (ground_truth > 0).any(dim=0)
        sample_mask = (ground_truth > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = ground_truth.masked_fill(ground_truth <= 0, NEG_INFINITE).masked_fill(ground_truth > 0, float(0.0))
        plogit_class = torch.logsumexp(-predictions / self.temperature_pos + pmask, dim=0).mul(self.temperature_pos)[class_mask]
        plogit_sample = torch.logsumexp(-predictions / self.temperature_pos + pmask, dim=1).mul(self.temperature_pos)[sample_mask]

        nmask = ground_truth.masked_fill(ground_truth != 0, NEG_INFINITE).masked_fill(ground_truth == 0, float(0.0))
        nlogit_class = torch.logsumexp(predictions / self.temperature_neg + nmask, dim=0).mul(self.temperature_neg)[class_mask]
        nlogit_sample = torch.logsumexp(predictions / self.temperature_neg + nmask, dim=1).mul(self.temperature_neg)[sample_mask]

        return (
            torch.nn.functional.softplus(nlogit_class + plogit_class).mean()
            + torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()
        )
