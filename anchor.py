# -*- coding: utf-8 -*-
"""
Created on 2018/10/17 15:30

@author: royce.mao

根据IOU计算生成anchors的过程，按既定比例生成正、负样本anchors

# np.tile：复制数组本身
# np.repeat：复制数组元素
# np.meshgrid：数组行、列复制
# np.vstack：数组堆叠
# np.ravel()：高维数组打平为一维
# np.stack：数组里面的元素堆叠
"""
from overlap import overlap
import numpy as np

def anchors_generation(base_size=None, ratios=None, scales=None):
    """
    根据base_size、ratios、scales3个指标生成(x1, y1, x2, y2)格式存储的feature map上的基准anchors
    :param base_size: 
    :param ratios: 
    :param scales: 
    :return: 
    """
    if base_size is None:
        base_size = 10
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0/3.0), 2 ** (2.0/3.0)])
    # 能够生成的anchors数量等于长宽比数量✖尺寸缩放数量
    anchors_num = len(ratios)*len(scales) # 3✖3=9个anchors
    # 初始化anchors存储格式
    anchors = np.zeros((anchors_num, 4)) # 9✖4维度的anchor取值
    # 赋值缩放尺寸
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T # scales.dim<(2, len(ratios)).dim，所以复制增加到2行，每行重复复制3次
    # print(anchors)
    # 计算anchors的3个缩放基准面积（直接scale_base_size✖scale_base_size）
    areas = anchors[:, 2] * anchors[:, 3]
    # print(areas)
    # 设置长宽比（面积不变，改变长宽比）
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    # print(anchors[:, 0::2])
    # print(anchors[:, 1::2])
    # 将anchor转换为(x1, y1, x2, y2)表示形式（以锚点为中心原点）
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T   # 第1、3列，x_ctr-0.5w, w-0.5w
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T   # 第2、4列，y_ctr-0.5h, h-0.5h
    # print(anchors)
    # 一个锚点上的9个anchors都用以锚点为原点的坐标表示了出来
    return anchors

def sliding_anchors_all(shape, stride, anchors):
    """
    假设shape = [256, 512]，stride = [32, 32]
    根据feature map的大小、步长以及基准anchors，滑动窗口，生成所有anchors映射到原图的坐标
    :param shape: 
    :param stride: 
    :param anchors: 
    :return: 
    """
    # 滑动窗口隐射到原图的中心点（相邻间距由1变为stride步长：32）
    sliding_x = (np.arange(0, shape[1]) + 0.5) * stride[1] # len=512
    sliding_y = (np.arange(0, shape[0]) + 0.5) * stride[0] # len=256
    # shift_x（256✖512，256行数值相同），shift_y（256✖512,512列数值相同）
    shift_x, shift_y = np.meshgrid(sliding_x, sliding_y)
    # shifts表示4个坐标值偏移中心点的偏移量? 维度为（256✖512，4）
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()
    # print(shifts)
    # 使用numpy的广播将 anchor (1, A, 4)和偏移anchor中心点(K, 1, 4) 相加，最终得到(K, A, 4)，然后再reshape (K*A, 4)
    A = anchors.shape[0] # 每个锚点基准anchors数量
    K = shifts.shape[0] # feature map 上锚点的数量
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    # print(anchors.reshape((1, A, 4))) # [0]长度为9的3维数组，基准anchor坐标
    # print("========================")
    # print(shifts.reshape((1, K, 4))) # [0]长度为256✖512的3维数组
    # print("========================")
    # print(all_anchors) # [0]长度为256✖512，[0][n]长度为9的3维数组
    all_anchors = all_anchors.reshape((K * A, 4))
    print(all_anchors)
    return all_anchors

def pos_neg_iou(pos_overlap, neg_overlap, all_anchors, GT):
    """
    计算all_anchors与给定的GT之间的overlaps，并为每个anchor匹配IOU值最高的gt（IOU值），根据阈值挑选正、负样本
    :param pos_overlap: 
    :param neg_overlap: 
    :param all_anchors: 
    :return: 
    """
    # IOU值计算
    overlaps = overlap(GT.astype(np.float64), all_anchors.astype(np.float64))
    # all_anchors中每个anchor最佳的IOU值与其对应的GT索引
    argmax_iou_index = np.argmax(overlaps.T, axis=1) # （1维数组）按行返回每个anchor对应的IOU值最高的GT索引
    # print(argmax_iou_index)
    argmax_iou_anchor = overlaps.T[np.arange(len(overlaps[0])), argmax_iou_index] # （1维数组）返回每个anchors最好的IOU值
    # print(argmax_iou_anchor)
    '''
    # 寻找每个GT最佳的anchor对应的index与IOU值
    argmax_iou_index2 = np.argmax(overlaps, axis=1)
    print(argmax_iou_index2)
    argmax_iou_anchor2 = overlaps[np.arange(len(overlaps.T[0])), argmax_iou_index2]
    print(argmax_iou_anchor2)
    # 如果某个GT对应最佳anchor的IOU等于0，说明GT标注完全与所有anchors都不相交，训练标注数据无意义
    # 可做判断，如果所有GT对应最佳anchor的IOU均高于阈值，继续；如果某个GT的最佳anchor的IOU低于阈值（该GT找不到正样本），取之前计算的IOU最高的anchor作为正样本
    '''
    # 提取正、负样本anchors的索引
    pos_inds = (argmax_iou_anchor >= pos_overlap)
    pos_index = ([i for i, x in enumerate(pos_inds) if x == True]) # IOU值高于0.1的正样本索引
    # print(pos_index)
    neutral_inds = (argmax_iou_anchor > neg_overlap) & ~pos_inds # IOU值介于0.05-0.1之间的中性样本
    neutral_index = ([i for i, x in enumerate(neutral_inds) if x == True])
    # print(neutral_index)
    neg_inds = (argmax_iou_anchor <= neg_overlap)
    neg_index = ([i for i, x in enumerate(neg_inds) if x == True]) # 所有负样本
    # 根据索引提取正、负、中性样本
    pos_sample = np.array([all_anchors[index] for index in pos_index])
    neutral_sample = np.array([all_anchors[index] for index in neutral_index])
    neg_sample = np.array([all_anchors[index] for index in neg_index])
    print(pos_sample)
    return pos_sample, neutral_sample, neg_sample


if __name__ == "__main__":
    shape = [224, 224] # shape = [256, 512]
    stride = [16, 16] # stride = [32, 32]
    GT = np.array(([10, 15, 20, 25], [200, 250, 210, 260]))
    pos_overlap = 0.1
    neg_overlap = 0.05
    anchors = anchors_generation()
    all_anchors = sliding_anchors_all(shape, stride, anchors)
    pos_neg_iou(pos_overlap, neg_overlap, all_anchors, GT)
