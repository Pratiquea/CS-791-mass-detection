#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs


class YOLOXCvAM(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x_l_cc, x_r_cc, x_l_mlo, x_r_mlo, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs_l_cc, fpn_outs_r_cc, fpn_outs_l_mlo, fpn_outs_r_mlo = self.backbone(x_l_cc, x_r_cc, x_l_mlo, x_r_mlo)

        if self.training:
            assert targets is not None
            targets_l_cc = targets[0]
            targets_r_cc = targets[1]
            targets_l_mlo = targets[2]
            targets_r_mlo = targets[3]

            loss_l_cc, iou_loss_l_cc, conf_loss_l_cc, \
            cls_loss_l_cc, l1_loss_l_cc, num_fg_l_cc = self.head( \
                fpn_outs_l_cc, targets_l_cc, x_l_cc )

            loss_r_cc, iou_loss_r_cc, conf_loss_r_cc, \
            cls_loss_r_cc, l1_loss_r_cc, num_fg_r_cc = self.head( \
                fpn_outs_r_cc, targets_r_cc, x_r_cc )
            
            loss_l_mlo, iou_loss_l_mlo, conf_loss_l_mlo, \
            cls_loss_l_mlo, l1_loss_l_mlo, num_fg_l_mlo = self.head( \
                fpn_outs_l_mlo, targets_l_mlo, x_l_mlo )

            loss_r_mlo, iou_loss_r_mlo, conf_loss_r_mlo, \
            cls_loss_r_mlo, l1_loss_r_mlo, num_fg_r_mlo = self.head( \
                fpn_outs_r_mlo, targets_r_mlo, x_r_mlo )
            

            outputs = {
                "total_loss_l_cc": loss_l_cc,
                "iou_loss_l_cc": iou_loss_l_cc,
                "l1_loss_l_cc": l1_loss_l_cc,
                "conf_loss_l_cc": conf_loss_l_cc,
                "cls_loss_l_cc": cls_loss_l_cc,
                "num_fg_l_cc": num_fg_l_cc,
                "total_loss_r_cc": loss_r_cc,
                "iou_loss_r_cc": iou_loss_r_cc,
                "l1_loss_r_cc": l1_loss_r_cc,
                "conf_loss_r_cc": conf_loss_r_cc,
                "cls_loss_r_cc": cls_loss_r_cc,
                "num_fg_r_cc": num_fg_r_cc,
                "total_loss_l_mlo": loss_l_mlo,
                "iou_loss_l_mlo": iou_loss_l_mlo,
                "l1_loss_l_mlo": l1_loss_l_mlo,
                "conf_loss_l_mlo": conf_loss_l_mlo,
                "cls_loss_l_mlo": cls_loss_l_mlo,
                "num_fg_l_mlo": num_fg_l_mlo,
                "total_loss_r_mlo": loss_r_mlo,
                "iou_loss_r_mlo": iou_loss_r_mlo,
                "l1_loss_r_mlo": l1_loss_r_mlo,
                "conf_loss_r_mlo": conf_loss_r_mlo,
                "cls_loss_r_mlo": cls_loss_r_mlo,
                "num_fg_r_mlo": num_fg_r_mlo,
            }
        else:
            output_l_cc = self.head(fpn_outs_l_cc)
            output_r_cc = self.head(fpn_outs_r_cc)
            output_l_mlo = self.head(fpn_outs_l_mlo)
            output_r_mlo = self.head(fpn_outs_r_mlo)
            outputs = [output_l_cc, output_r_cc, output_l_mlo, output_r_mlo]

        return outputs