# -*- coding: utf-8 -*-
"""
Created on 2018/10/23 09:47

@author: royce.mao

第1阶段，rpn_loss_cls 和 rpn_loss_regr (还需要修改)
"""

from keras import backend as K
import tensorflow as tf
import numpy as np

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

epsilon = 1e-4

# Sigmoid loss对于多类别，也只返回属于前景类的概率（0，9，0）
def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return lambda_rpn_class * K.sum(
            y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(
            epsilon + y_true[:, :, :, :num_anchors]) # y_is_box_valid -> 属于前景、背景类标为1，忽略类标为0

    return rpn_loss_cls_fixed_num


# Smooth L1 loss（36，0，0）
def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, :, 4*num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4*num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :, :4*num_anchors])  # np.repeat(y_rpn_overlap, 4, axis=1) -> 前景类标为1，其余类标为0

    return rpn_loss_regr_fixed_num

