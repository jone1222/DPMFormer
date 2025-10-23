# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import get_class_weight, weight_reduce_loss



def kl_divergence(p, q, eps=1e-7):
    """
    Calculate the KL divergence between two distributions.
    We assume the distributions are along the last dimension (-1).
    """
    # Calculate KL Divergence
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    p = p / p.sum(-1, keepdim=True)
    q = q / q.sum(-1, keepdim=True)

    kl_div = p * (p.log() - q.log())
    # Sum over the last dimension to get the total KL divergence per example in the batch
    return kl_div.sum(-1)

def js_divergence(p, q, eps=1e-7):
    """
    Calculate the Jensen-Shannon Divergence between two distributions.
    """
    # # Ensure distributions sum to 1 over the dim of interest (in case they are not normalized)
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(-1, keepdim=True)
    
    # Calculate the M distribution
    m = 0.5 * (p + q)
    
    # Calculate the KL divergences
    kl_p_m = kl_divergence(p, m, eps)
    kl_q_m = kl_divergence(q, m, eps)
    
    # Calculate the JSD
    jsd = 0.5 * (kl_p_m + kl_q_m)
    
    return jsd.mean()

def loss_query_target_class_consistency(outputs):
    preds = outputs

    B, Q, C = preds.shape

    org_pred = preds[:B//2].reshape(-1, C)
    aug_pred = preds[B//2:].reshape(-1, C)

    org_pred = F.softmax(org_pred[..., :-1], dim=-1)
    aug_pred = F.softmax(aug_pred[..., :-1], dim=-1)

    loss = js_divergence(org_pred, aug_pred)

    return loss


@LOSSES.register_module(force=True)
class ClassConsistencyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ce',
                 avg_non_ignore=False):
        super(ClassConsistencyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        
        self.cls_consistency_criterion = loss_query_target_class_consistency
        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        # Note: for BCE loss, label < 0 is invalid.
        loss_cls_consistency = self.loss_weight * self.cls_consistency_criterion(cls_score)

        return loss_cls_consistency

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
