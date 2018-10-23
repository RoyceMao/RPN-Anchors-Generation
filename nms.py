# -*- coding: utf-8 -*-
"""
Created on 2018/10/22 09:14

@author: royce.mao

根据rpn_cls_score对proposals的评分对regions of interest，进行非极大抑制，进一步削减训练样本。
"""

# coding:utf-8
from overlap import overlap
import numpy as np

def nms_plan(scores, boxes, threshold = 0.7, class_sets = None):
    """
    根据类别判断RPN分类、回归预测后的bbox集合（未完成）
    post-process the results of im_detect
    :param scores: N * (K * 4) numpy
    :param boxes: N * K numpy
    :param class_sets: e.g. CLASSES = ('__background__','person','bike','motorbike','car','bus')
    :return: a list of K-1 dicts, no background, each is {'class': classname, 'dets': None | [[x1,y1,x2,y2,score],...]}
    """
    num_class = scores.shape[1] if class_sets is None else len(class_sets)
    assert num_class * 4 == boxes.shape[1],\
        'Detection scores and boxes dont match'
    class_sets = ['class_' + str(i) for i in range(0, num_class)] if class_sets is None else class_sets

    res = []
    for ind, cls in enumerate(class_sets[1:]):
        ind += 1 # skip background
        cls_boxes =  boxes[:, 4*ind : 4*(ind+1)]
        cls_scores = scores[:, ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, thresh=0.3)
        dets = dets[keep, :]
        dets = dets[np.where(dets[:, 4] > threshold)]
        r = {}
        if dets.shape[0] > 0:
            r['class'], r['dets'] = cls, dets
        else:
            r['class'], r['dets'] = cls, None
        res.append(r)
    return res


def nms(bbox, thresh):
    """
    局部非极大抑制
    :param bbox: 
    :param thresh:
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
