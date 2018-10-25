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
from overlap import overlap
from voc_data import voc_final
from net_layers import resnet50, roi_pooling_conv, roi_pooling_layer
import numpy as np

def proposal_to_roi_plan(proposals, gt_boxes, classifier_min_overlap, classifier_max_overlap):
    """
    原图nms后生成的proposals做具体类别的cls标注，与坐标形式转换：[x1, y1, x2, y2]转换为[x1, y1, w, h]
    :param proposals: nms输出
    :param gt_boxes: gt
    :return: 
    """
    # cls 标注准备
    max_index = []
    max_iou = []
    iou = overlap(proposals, np.array((gt_boxes[:, :4]), dtype=np.float32))
    # print(np.where(iou[0] == np.max(iou[0]))[0])
    for i in range(len(iou)):
        ## 找到每个proposal最大的iou，以及对应的gt索引
        max_iou.append(np.max(iou[i]))
        max_index.append(np.where(iou[i] == np.max(iou[i]))[0])
    # 坐标
    w = proposals[:,2] - proposals[:,0]
    h = proposals[:,3] - proposals[:,1]
    proposals[:,2] = w
    proposals[:,3] = h
    # print(proposals)
    # max_iou与overlap阈值的比较,区分第2阶段的正、负样本，并做cls的标注
    pos_index = np.where(np.array(max_iou) >= classifier_max_overlap)
    # print(pos_gt_index)
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
    return rois, cls_target

def proposal_to_roi(rois_pic, stride):
    """
    原始图像RoI到feature map上RoI的映射，并计算第2阶段的回归目标
    :param proposals: 
    :param gt_boxes: 
    :param classifier_min_overlap: 
    :param classifier_max_overlap: 
    :return: 
    """
    rois_map = rois_pic
    # feature mapping，将ROI映射到feature map对应位置
    rois_map[:,0] = rois_pic[:, 0] / stride[0]
    rois_map[:,1] = rois_pic[:, 1] / stride[0]
    rois_map[:,2] = rois_pic[:, 2] / stride[1]
    rois_map[:,3] = rois_pic[:, 3] / stride[1]
    # 计算第2阶段的回归目标
    # regression_target ===
    return rois_map, regression_target

def resnet50_roi_pooling(rois_map, cls_target, regr_target):
    """
    resnet50的基础特征提取网络 + roi pooling conv结构 + cls_layer、regr_layer
    :param rois: feature map对应的rois
    :param pooling_size: 池化后的尺寸
    :param num_rois:
    :return: 
    """
    base_layer = resnet50()
    out_roi_pool = roi_pooling_layer(base_layer, rois_map, pooling_size=14, num_rois=len(rois_map))
    # conv_block_td、classifier_layers、out_class、out_regr
    # ===

if __name__ == "__main__":
    # proposal_to_roi函数测试
    proposals = np.array(([1,2,3.5,4.5],[4,5,6.5,7.5],[8,9,10.5,11.5],[12,13,14.5,15.5],[16,17,18,19]))
    gt_boxes = np.array(([1,2,3,4,'person'],[4,5,6,7,'dog'],[8,9,10,11,'sheep'],[12,13,14,15,'bow']))
    classifier_min_overlap = 0.0
    classifier_max_overlap = 0.25
    rois, cls = proposal_to_roi_plan(proposals, gt_boxes, classifier_min_overlap, classifier_max_overlap)
    print('ROIs：\n{}'.format(rois))
    print('ROIs标注：\n{}'.format(cls))
    # roi_pooling函数测试

