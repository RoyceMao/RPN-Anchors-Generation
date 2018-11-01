# -*- coding: utf-8 -*-
"""
Created on 2018/10/24 14:58

@author: royce.mao

RoI Pooling，把所有RoIs映射到特征图上，根据pooling_size，进行大小固定为（w * h）的矩形框Pooling。
输入：
  1）与rpn共享的share_conv，resnet50_rpn第5个block之后的feature map；
  2）rpn经过regr、nms之后的RoIs输出；
输出：
  与输入RoIs数量相同，但大、小固定为（w * h）且映射了feature的RoIs
"""
from nms import nms
from overlap import overlap_gt
from voc_data import voc_final
from net_layers import resnet50, roi_pooling_conv, roi_pooling_layer
import numpy as np

def cls_target(proposals, gt_boxes, classifier_min_overlap, classifier_max_overlap):
    """
    做具体类别的cls标注，与坐标形式转换：[x1, y1, x2, y2]转换为[x1, y1, w, h]
    :param proposals: proposals窗口以及对应的feature map特征
    :param gt_boxes: gt
    :return: 
    """
    # cls 标注准备
    max_index = []
    max_iou = []
    iou = overlap_gt(proposals, np.array((gt_boxes[:, :4]), dtype=np.float32))
    # print(np.where(iou[0] == np.max(iou[0]))[0])
    for i in range(len(iou)):
        ## 找到每个proposal最大的iou，以及对应的gt索引
        max_iou.append(np.max(iou[i]))
        max_index.append(np.where(iou[i] == np.max(iou[i]))[0]) # 某个proposal对应最大iou的gt的index
    # 坐标
    w = proposals[:,2] - proposals[:,0]
    h = proposals[:,3] - proposals[:,1]
    proposals[:,2] = w
    proposals[:,3] = h
    # print(proposals)
    # max_iou与overlap阈值的比较,区分第2阶段的正、负样本，并做cls的标注
    pos_index = np.where(np.array(max_iou) >= classifier_max_overlap) # proposals的index
    # print(pos_index[0])
    pos_cls = gt_boxes[[max_index[x][0] for x in pos_index[0]], 4]
    # print(pos_cls)
    hard_neg_index = [np.where(max_iou == iou)[0] for iou in max_iou if classifier_min_overlap <= iou < classifier_max_overlap]
    # 对应合并
    cls_target = list(np.zeros(len(proposals)))
    for hard_inds in [x[0] for x in hard_neg_index]:
        cls_target[hard_inds] = "bg"
    for i, x in enumerate(pos_index[0]):
        cls_target[x] = pos_cls[i]
    rois = proposals
    return rois, cls_target, pos_index, max_index

def regr_target(rois, gt_boxes, pos_index, max_index):
    """
    做具体目标框回归目标与回归修正的计算公式逻辑
    :param proposals: rois窗口以及对应的feature map特征
    :param gt_boxes: gt
    :param pos_index: 正样本的下标索引
    :param max_index: 正样本对应的最佳gt的下标索引
    :return: 
    """
    # 所有正样本对应的最佳gt的中心点
    gt_x_center = (gt_boxes[[max_index[x][0] for x in pos_index[0]], 0].astype(np.float64) + gt_boxes[[max_index[x][0] for x in pos_index[0]], 2].astype(np.float64)) / 2.0
    gt_y_center = (gt_boxes[[max_index[x][0] for x in pos_index[0]], 1].astype(np.float64) + gt_boxes[[max_index[x][0] for x in pos_index[0]], 3].astype(np.float64)) / 2.0
    # 所有正样本本身的中心点
    x_center = rois[pos_index[0], 0] + rois[pos_index[0], 2] / 2.0
    y_center = rois[pos_index[0], 1] + rois[pos_index[0], 3] / 2.0
    # 偏移量、缩放量（回归目标）
    dx = (gt_x_center - x_center) / rois[pos_index[0], 2]
    dy = (gt_y_center - y_center) / rois[pos_index[0], 3]
    dw = np.log((gt_boxes[[max_index[x][0] for x in pos_index[0]], 1].astype(np.float64) - gt_boxes[[max_index[x][0] for x in pos_index[0]], 0].astype(np.float64)) / rois[pos_index[0], 2])
    dh = np.log((gt_boxes[[max_index[x][0] for x in pos_index[0]], 3].astype(np.float64) - gt_boxes[[max_index[x][0] for x in pos_index[0]], 2].astype(np.float64)) / rois[pos_index[0], 3])
    # 计算回归修正
    x_target_center = dx * rois[pos_index[0], 2] + x_center
    y_target_center = dy * rois[pos_index[0], 3] + y_center
    w_target = np.exp(dw) * rois[pos_index[0], 2]
    h_target = np.exp(dh) * rois[pos_index[0], 3]
    x_target = x_target_center - w_target / 2.0
    y_target = y_target_center - h_target / 2.0
    return np.stack([x_target, y_target, w_target, h_target]).T, np.stack([dx, dy, dw, dh]).T


def proposal_to_roi(rois_pic, stride):
    """
    原图RoI到feature map上RoI的坐标映射
    :param rois_pic: 原图的RoIs
    :param stride: 步长
    :return: 
    """
    rois_map = rois_pic
    # feature mapping，将ROI映射到feature map对应位置
    rois_map[:,0] = rois_pic[:, 0] / stride[0]
    rois_map[:,1] = rois_pic[:, 1] / stride[0]
    rois_map[:,2] = rois_pic[:, 2] / stride[1]
    rois_map[:,3] = rois_pic[:, 3] / stride[1]
    return rois_map


if __name__ == "__main__":
    # cls_target函数数学逻辑测试
    proposals = np.array(([1,2,3.5,4.5],[4,5,6.5,7.5],[8,9,10.5,11.5],[12,13,14.5,15.5],[16,17,18,19]))
    gt_boxes = np.array(([1,2,3,4,'person'],[4,5,6,7,'dog'],[8,9,10,11,'sheep'],[12,13,14,15,'bow']))
    classifier_min_overlap = 0.0
    classifier_max_overlap = 0.25
    rois, cls, pos_index, max_index = cls_target(proposals, gt_boxes, classifier_min_overlap, classifier_max_overlap)
    print('ROIs：\n{}'.format(rois))
    print('ROIs分类目标：\n{}'.format(cls))
    # regr_target函数数学逻辑测试
    revise, shift = regr_target(rois, gt_boxes, pos_index, max_index)
    print('正样本ROIs回归目标：\n{}'.format(shift))
    print('正样本ROIs回归修正：\n{}'.format(revise))
    # 最终分类、回归测试


