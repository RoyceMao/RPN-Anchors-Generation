# -*- coding: utf-8 -*-
"""
Created on 2018/10/23 09:47

@author: royce.mao

rpn_loss_cls 和 rpn_loss_regr
"""

from keras import backend as K
import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


# inds是标注为0、1的所有正、负样本，用于分类
def rpn_loss_cls():
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return lambda_rpn_class * K.sum(
            y_true[:, :, :] * K.binary_crossentropy(y_pred[:, :, :], y_true[:, :, :])) / K.sum(
            epsilon + y_true[:, :, :])

    return rpn_loss_cls_fixed_num


# 　pos_inds是标注为1的所有正样本，用于回归
def rpn_loss_regr():
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, :] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
        return lambda_rpn_regr * K.sum(
            y_true[:, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :])

    return rpn_loss_regr_fixed_num


# 合并
def Loss():
    rpn_loss_cls_fixed_num = rpn_loss_cls()
    rpn_loss_regr_fixed_num = rpn_loss_regr()
    return rpn_loss_cls_fixed_num, rpn_loss_regr_fixed_num
