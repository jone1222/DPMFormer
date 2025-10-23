import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize
from mmcv.runner import ModuleList, force_fp32

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class TextIdentityHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, **kwargs):
        super(TextIdentityHead, self).__init__(
            input_transform=None, **kwargs)
        self.conv_seg = None

    def forward(self, inputs):
        return inputs
    
    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, seg_weight)
        return losses
    
    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_weight=None):

        """Compute segmentation loss."""
        loss = dict()
        loss['loss_text_aug_consistency'] = self.loss_decode(
            seg_logit,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        return loss
