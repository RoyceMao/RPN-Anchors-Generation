# -*- coding: utf-8 -*-
"""
Created on 2018/10/29 11:39

@author: royce.mao

第2阶段，frcnn_loss_cls 和 frcnn_loss_regr
"""
from keras import backend as K
from keras.objectives import categorical_crossentropy

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def class_loss_cls(y_true, y_pred):
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))


def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num
