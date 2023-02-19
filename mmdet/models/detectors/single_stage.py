import torch
import torch.nn as nn
from torch.autograd import Function

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

class CheckpointFunction(Function):
    @staticmethod
    def forward(ctx, x):
        assert not x.is_checkpoint(),"x must not a checkpoint tensor"
        return x.detach().checkpoint()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.decheckpoint()

class Checkpoint(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return CheckpointFunction.apply(x)

class DecheckpointFunction(Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_checkpoint(),"x must be a checkpoint tensor"
        return x.decheckpoint()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.checkpoint()

class Decheckpoint(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return DecheckpointFunction.apply(x)

class Wrapper(nn.Module):
    def __init__(self, op):
        super().__init__()
        a = 5 + 4 + 4
        b = 10
        self.decheckpoint = [Decheckpoint() for _ in range(a)]
        self.op = op
        self.checkpoint = [Checkpoint() for _ in range(b)]
        
    
    def forward(self, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore):
        origin_x = tuple([self.decheckpoint[i](x[i]) for i in range(5)])
        # origin_gt_bboxes = [self.decheckpoint[i + 5](gt_bboxes[i]) for i in range(4)]
        # origin_gt_labels = [self.decheckpoint[i + 9](gt_labels[i]) for i in range(4)]
        origin_gt_bboxes = gt_bboxes
        origin_gt_labels = gt_labels
        origin_output = self.op(origin_x, img_metas, origin_gt_bboxes, origin_gt_labels, gt_bboxes_ignore)
        checkpoint_output = {
            'loss_cls': [self.checkpoint[i](origin_output['loss_cls'][i]) for i in range(5)],
            'loss_bbox': [self.checkpoint[i + 5](origin_output['loss_bbox'][i]) for i in range(5)],
        }
        return checkpoint_output


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        # self.wrap = Wrapper(self.bbox_head.forward_train)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        self.test_memory_usage = torch.cuda.memory_allocated()
        self.input_shape = img.shape
        if hasattr(self, 'dc_manager') and self.dc_manager:
            self.dc_manager.set_input_size(img.shape[-1] * img.shape[-2])
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        assert len(x) == 5
        # x = tuple([tmp.decheckpoint() for tmp in x])
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        # losses = self.wrap(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
