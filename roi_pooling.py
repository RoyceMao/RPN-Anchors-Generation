# -*- coding: utf-8 -*-
"""
Created on 2018/10/24 14:58

@author: royce.mao

RoI Pooling，把各个proposal映射到特征图上，得到RoIs,并进行大小固定为（w * h）的矩形框Pooling。
输入：
  1）与rpn共享的share_conv，resnet50_rpn第5个block之后的feature map；
  2）rpn经过nms之后的proposals输出；
输出：
  与输入proposals数量相同，大、小固定（w * h）的RoIs
"""
from nms import nms
from overlap import overlap
from train_rpn import resnet50_rpn
import numpy as np

def proposal_to_roi(proposals, gt_boxes, classifier_min_overlap, classifier_max_overlap):
    """
    nms后生成的proposals做具体类别的cls标注，与坐标形式转换：[x1, y1, x2, y2]转换为[x1, y1, w, h]
    :param proposals: nms输出
    :param gt_boxes: gt
    :return: 
    """
    # cls 标注准备
    max_index = []
    max_iou = []
    iou = overlap(proposals, gt_boxes[:, :4])
    for i in range(len(iou)):
        ## 找到每个proposal最大的iou，以及对应的gt索引
        max_iou.append(np.max(iou[i]))
        max_index.append(np.where(iou[i] == np.max(iou[i])))
    # 坐标
    w = proposals[:,2] - proposals[:,0]
    h = proposals[:,3] - proposals[:,1]
    proposals[:,2] = w
    proposals[:,3] = h
    print(proposals)
    # max_iou与overlap阈值的比较,区分2阶段的正、负样本，并做cls的标注
    pos_index = np.where(max_iou >= classifier_max_overlap)
    pos_cls = gt_boxes[max_index[pos_index], 4]
    hard_neg_index = np.where(classifier_min_overlap <= max_iou < classifier_max_overlap)
    hard_neg_cls = 'bg'
    # 对应合并
    cls = np.zeros(len(proposals))
    cls[hard_neg_index] = 'bg'
    for i, x in enumerate(pos_index):
        cls[x] = pos_cls[i]
    rois = np.concatenate(proposals, cls)
    return rois


def feature_mapping(proposals, feature_map, stride):
    """
    
    :param proposals: 
    :param feature_map: 
    :param stride: 
    :return: 
    """
    # proposals映射回feature map的坐标情况
    x1_map = proposals[:, 0] / stride[0]
    x2_map = proposals[:, 2] / stride[0]
    y1_map = proposals[:, 1] / stride[1]
    y2_map = proposals[:, 3] / stride[1]
    # Pooling


def roi_pooling():
