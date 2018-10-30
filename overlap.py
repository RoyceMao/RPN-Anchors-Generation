# -*- coding: utf-8 -*-
"""
Created on 2018/10/17 14:30

@author: royce.mao

# 计算两组boxes的IOU值，返回(N,M)维数组，N是GT数量，M是anchor数量
# 其中，anchor_box的存储方式为numpy.array(x1,y1,x2,y2)，gt_box的存储方式为numpy.array(x1,x2,y1,y2)
"""
import numpy as np

def overlap_gt(boxes,query_boxes):
     # 1维输入统一扩展为2维
     if len(boxes.shape) == 1:
        boxes = np.reshape(boxes, (-1, len(boxes)))
     M = boxes.shape[0] # ground truth boxes个数
     N = query_boxes.shape[0] # 待检测overlap的anchor boxes个数
     overlaps = np.zeros((M, N))
     # 循环计算每个anchor box与所有 ground truth boxes的IOU
     for n in range(N):
         # 计算每个anchor box的面积
        box_area = (
            (query_boxes[n, 2] - query_boxes[n, 0] + 1) *
            (query_boxes[n, 3] - query_boxes[n, 1] + 1)
        )
        for m in range(M):
            iw = (
                  min(boxes[m, 1], query_boxes[n, 2]) -
                  max(boxes[m, 0], query_boxes[n, 0]) + 1
             )
            if iw > 0: # 大于零说明水平方向上相交
                ih = (
                    min(boxes[m, 3], query_boxes[n, 3]) -
                    max(boxes[m, 2], query_boxes[n, 1]) + 1
                 )
                if ih > 0: # 大于零说明垂直方向上相交
                    # ua是两个boxes面积的并集
                    ua = np.float64(
                        (boxes[m, 1] - boxes[m, 0] + 1) *
                        (boxes[m, 3] - boxes[m, 2] + 1) +
                         box_area - iw * ih
                    )
                    # iw * ih是两个boxes面积的交集
                    overlaps[m, n] = iw * ih / ua
     return  overlaps


def overlap(boxes, query_boxes):
    # 1维输入统一扩展为2维
    if len(boxes.shape) == 1:
        boxes = np.reshape(boxes, (-1, len(boxes)))
    M = boxes.shape[0]  # ground truth boxes个数
    N = query_boxes.shape[0]  # 待检测overlap的anchor boxes个数
    overlaps = np.zeros((M, N))
    # 循环计算每个anchor box与所有 ground truth boxes的IOU
    for n in range(N):
        # 计算每个anchor box的面积
        box_area = (
            (query_boxes[n, 2] - query_boxes[n, 0] + 1) *
            (query_boxes[n, 3] - query_boxes[n, 1] + 1)
        )
        for m in range(M):
            iw = (
                min(boxes[m, 2], query_boxes[n, 2]) -
                max(boxes[m, 0], query_boxes[n, 0]) + 1
            )
            if iw > 0:  # 大于零说明水平方向上相交
                ih = (
                    min(boxes[m, 3], query_boxes[n, 3]) -
                    max(boxes[m, 1], query_boxes[n, 1]) + 1
                )
                if ih > 0:  # 大于零说明垂直方向上相交
                    # ua是两个boxes面积的并集
                    ua = np.float64(
                        (boxes[m, 2] - boxes[m, 0] + 1) *
                        (boxes[m, 3] - boxes[m, 1] + 1) +
                        box_area - iw * ih
                    )
                    # iw * ih是两个boxes面积的交集
                    overlaps[m, n] = iw * ih / ua
    return overlaps

if __name__ == "__main__":
    boxes =  np.ndarray(shape=(4,), dtype=int, buffer=np.array([1,2,3,4]), offset=0, order="C")
    print(boxes)
    query_boxes = np.ndarray(shape=(3,4), dtype=int, buffer=np.array([1,2,3,4,5,6,7,8,9,10,11,12]), offset=0, order="C")
    print(query_boxes)
    print(overlap(boxes,query_boxes).T)
