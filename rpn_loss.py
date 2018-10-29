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

# Sigmoid loss对于多类别，也只返回属于前景类的概率
def rpn_loss_cls():
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return lambda_rpn_class * K.sum(
            y_true[:, :, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, :])) / K.sum(
            epsilon + y_true[:, :, :, :]) # y_is_box_valid

    return rpn_loss_cls_fixed_num


# Smooth L1 loss
def rpn_loss_regr():
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, :, :] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :, :])  # np.repeat(y_rpn_overlap, 4, axis=1)

    return rpn_loss_regr_fixed_num

