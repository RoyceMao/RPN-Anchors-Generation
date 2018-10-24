# -*- coding: utf-8 -*-
"""
Created on 2018/10/22 09:14

@author: royce.mao

根据rpn_cls_score对proposals的评分对regions of interest，进行非极大抑制，进一步削减训练样本。

# 注：overlap函数有修改，4个坐标的位置有变动，跑nms时需要改回来
"""

# coding:utf-8
from overlap import overlap
import numpy as np


def nms(bbox, thresh, max_boxes):
    """
    局部非极大抑制
    :param bbox: 1张图（1个batch）的所有proposals
    :param thresh:
    :return: 
    """
    proposals = bbox[:, :4] # bbox框
    scores = bbox[:, 4]  # bbox打分
    # bbox打分从大到小顺序排列，并返回各自的index,如[2,0,1,3,4]
    order = scores.argsort()[::-1]
    # 初始化keep
    keep = []
    # 计算临时最高分窗口与临时所有窗口的IOU
    while len(order) > 0:
        # 循环剔除不满足阈值条件的高重叠IOU proposals
        keep.append(order[0])
        iou = overlap(proposals[order[0], :], bbox[:,:4])
        inds = np.where(np.reshape(iou, (len(order))) > thresh)[0]
        bbox = np.delete(bbox, inds, 0) # 删除指定index对应的值
        order = [x for x in order if x not in inds] # 删除指定index
        # break条件
        if len(order) < 2 or len(keep) >= max_boxes:
            break
    boxes = proposals[keep]
    probs = scores[keep]
    return boxes, probs

if __name__ == "__main__":
    bbox = np.array(([1,2,3,4,10],[4,5,6,7,20],[4,5,6,7,40],[1,2,3.5,4.5,30]))
    thresh = 0.5
    a, b = nms(bbox, thresh, 4)
    print('剩余的proposals对应的ROIs：\n{}'.format(a))
    print('剩余的proposals对应的scores：\n{}'.format(b))