# -*- coding: utf-8 -*-
"""
Created on 2018/10/22 09:14

@author: royce.mao

根据rpn_cls_score对proposals的评分对regions of interest，进行非极大抑制，进一步削减训练样本。
"""

# coding:utf-8
from overlap import overlap
import numpy as np

def nms(bbox, thresh):
    """
    分类、回归后从属于相同GT的bbox，进行非极大抑制
    筛选出来的bbox
    :param dets: bbox的打分
    :param thresh: IOU阈值
    :return: 
    """
    scores = bbox[:, 4]  # bbox打分
    # bbox打分从大到小顺序排列，取各自的index
    order = scores.argsort()[::-1]
    # keep为最后保留的bbox对应的index
    keep = []
    # order[0]是当前分数最大的窗口的分数，肯定保留
    score_max_index = order[0]
    keep.append(score_max_index)
    # 计算窗口i与所有窗口的交叠部分的面积
    iou = overlap(np.reshape(bbox[score_max_index, :4], (-1, 4)), bbox[:, :4])
    inds = np.where(np.reshape(iou, (len(scores))) <= thresh)[0]
    for ind in inds:
        keep.append(ind)
    return keep

if __name__ == "__main__":
    bbox = np.array(([1,2,3,4,10],[4,5,6,7,20],[4,5,6,7,40],[1,2,3.5,4.5,30]))
    thresh = 0.5
    a = nms(bbox, thresh)
    print(a)
