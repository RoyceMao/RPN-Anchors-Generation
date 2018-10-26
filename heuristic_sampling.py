# -*- coding: utf-8 -*-
"""
Created on 2018/10/19 12:30

@author: royce.mao

有了voc标注数据和每张pic的候选anchors之后，考虑到正、负样本的极不均衡，需要进行1：3的启发式采样，生成proposals。
"""
from anchor import anchors_generation, sliding_anchors_all, pos_neg_iou
from voc_data import voc_final
import numpy as np
import keras

def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """
    计算anchor-GT对的边框回归的目标
    :param anchors: 一张图像所有的anchors, (x1,y1,x2,y2)
    :param gt_boxes: anchors对应的gt ,(x1,y1,x2,y2)
    :param mean:
    :param std:
    :return:
    """

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    # 计算长宽
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    # 计算回归目标(左上和右下坐标),没有长宽回归
    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets = targets.T

    # 标准化
    targets = (targets - mean) / std
    return targets


def anchor_targets_bbox(
    anchors,
    image_group,
    annotations_group,
    num_classes,
    pos_overlap,
    neg_overlap,
    class_mapping
):
    """ 
    生成RPN网络中一个batch边框分类和回归的目标
    """
    assert (len(image_group) == len(annotations_group)), "图片和标注数量不一致！"
    # batch_size设置为X张的图片数量
    batch_size = len(image_group)
    # 最后一位为标志位，-1:ignore, 0:negtive, 1:postive
    regression_batch = np.zeros((batch_size, anchors.shape[0], 4 + 1), dtype=keras.backend.floatx())
    # print(regression_batch)
    labels_batch = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=keras.backend.floatx())
    # print(labels_batch)
    boxes_batch = np.zeros((batch_size, anchors.shape[0], 5), dtype=keras.backend.floatx())
    # print(boxes_batch)

    # compute labels and regression targets
    for index, pic_name in enumerate(annotations_group.keys()):
        annotations_single = annotations_group[pic_name]
        # 用numpy把1✖n的一维数组，转化为n✖4的二维数组（n代表一张pic里面的GT数量）
        for i, gt_box in enumerate(annotations_single):
            a = gt_box[:4]
            a.append((class_mapping[gt_box[-1]]))
            if i == 0:
                annotations = np.array((a))
                annotations_nolabel = np.array((gt_box[:4]))
            else:
                annotations = np.vstack((annotations, np.array((a))))
                annotations_nolabel = np.vstack((annotations_nolabel, np.array((gt_box[:4]))))
        # print(annotations_nolabel)
        # print(annotations)
        # 正、负、中性样本的区分，以及每个anchor对应的最佳GT索引
        positive_indices, ignore_indices, argmax_overlaps_inds = pos_neg_iou(pos_overlap, neg_overlap, anchors, annotations_nolabel)
        # 全部初始化为忽略，再赋值正样本
        labels_batch[index, :, -1] = -1
        labels_batch[index, positive_indices, -1] = 1
        # 全部初始化为忽略，再赋值正样本
        regression_batch[index, :, -1] = -1
        regression_batch[index, positive_indices, -1] = 1
        # ===========================================================================
        # 计算边框回归的目标（所有anchors都计算，并非）
        annotations = annotations[argmax_overlaps_inds] # 每个anchor对应的最佳GT索引
        boxes_batch[index, ...] = annotations  # 记录类别
        # 计算目标类别，默认都是背景类，如[0,0,0,0,1,0,0]]
        labels_batch[index, positive_indices, annotations[positive_indices, 4].astype(int)] = 1  # 赋值标记位
        # 计算回归目标，与记录的类别无关
        regression_batch[index, :, :-1] = bbox_transform(anchors, annotations)
        # 按照1:3正、负样本比启发式采样
        postive_num = np.sum(labels_batch[index, :, -1] == 1)
        for i in np.random.randint(0, anchors.shape[0], 3 * postive_num):
            if not (labels_batch[index, :, -1]-1).all():
                labels_batch[index, i, -1] = 0  # 设为背景类
                regression_batch[index, i, -1] = 0
        # 忽略的
        labels_batch[index, ignore_indices, -1] = -1
        regression_batch[index, ignore_indices, -1] = -1
        # 越界边框标注为-1，并忽略
        anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
        indices = np.logical_or(anchors_centers[:, 0] >= ([224, 224])[1], anchors_centers[:, 1] >= ([224, 224])[0])
        labels_batch[index, indices, -1] = - 1
        regression_batch[index, indices, -1] = -1

    # 计算一个batch的图像，标注为前景、背景类的anchors数量之和
    num_anchors = np.sum(labels_batch[:, :, -1] == 1) + np.sum(labels_batch[:, :, -1] == 0)
    # 打印正负样本数量
    print("删除越界边框后1:3启发式采样：post_num:{},bg_num:{},ignore_num:{},proposals_num:{}".format(np.sum(labels_batch[:, :, -1] == 1),
                                                       np.sum(labels_batch[:, :, -1] == 0),
                                                       np.sum(labels_batch[:, :, -1] == -1),
                                                       num_anchors))
    # 提取batch中所有的正、负样本索引
    # pos_inds = (labels_batch[:,:,-1] == 1).ravel()
    inds = (labels_batch[:,:,-1] != -1).ravel()
    return labels_batch, regression_batch, num_anchors, inds

if __name__ == "__main__":
    # 准备voc的GT标注数据集
    data_path = "F:\\VOC2007"
    width = 224
    height = 224
    class_mapping, classes_count, all_images, all_annotations = voc_final(data_path)
    print(all_images)
    print(all_annotations)
    # all_annotations = np.array(value for value in dict_annotations.values())
    # 界定正、负样本的阈值边界
    pos_overlap = 0.5
    neg_overlap = 0.4
    # 一张图，一张图生成anchors并进行启发式采样
    anchors = anchors_generation()
    all_anchors = sliding_anchors_all([width, height], [1, 1], anchors)
    # 启发式采样
    labels_batch, regression_batch, num_anchors, inds = anchor_targets_bbox(all_anchors, all_images, all_annotations, len(classes_count), pos_overlap, neg_overlap, class_mapping)



