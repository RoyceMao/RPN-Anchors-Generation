# -*- coding: utf-8 -*-
"""
Created on 2018/10/22 14:00

@author: royce.mao

rpn网络针对proposals（1：3采样后）的binary前景背景评分，筛选出regions of interest，以及bbox regression回归。
"""
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed
from keras.models import Model
from anchor import anchors_generation, sliding_anchors_all, pos_neg_iou
from voc_data import voc_final
from PIL import Image
import numpy as np
import cv2

def rpn(num_anchors):
    input_tensor = Input(shape=(224, 224, 3))
    # relu的全连接层
    x = Convolution2D(224, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(input_tensor)
    # rpn_cls与rpn_regression的分支
    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    # summary
    cls_model = Model(inputs=input_tensor, outputs=x_class, name='cls')
    regr_model = Model(inputs=input_tensor, outputs=x_regr, name='regr')

    cls_model.summary()
    regr_model.summary()
    return cls_model, regr_model
'''
def rpn_cls(x_class):

def rpn_regression(x_regr):
'''

if __name__ == "__main__":
    # 准备voc的GT标注数据集
    data_path = "F:\\VOC2007"
    width = 224
    height = 224
    class_mapping, classes_count, all_images, all_annotations = voc_final(data_path, width, height)
    # 界定正、负样本的阈值边界
    pos_overlap = 0.5
    neg_overlap = 0.4
    # 一张图，一张图生成anchors并进行启发式采样
    anchors = anchors_generation()
    all_anchors = sliding_anchors_all([width, height], [1, 1], anchors)
    # 一张图的rpn网络测试
    img_path = "F:\\VOC2007\\JPEGImages\\000004.jpg"
    image = np.array(Image.open(img_path).convert("RGB"))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    # rpn_cls 和 rpn_regression
    cls_model, regr_model = rpn(48056)
    # 如何开始训练RPN网络呢？