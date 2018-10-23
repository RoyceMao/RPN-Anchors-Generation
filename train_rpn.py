# -*- coding: utf-8 -*-
"""
Created on 2018/10/22 14:00

@author: royce.mao

rpn网络针对proposals（1：3采样后）的binary前景背景评分，筛选出regions of interest，以及bbox regression回归。
"""
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from anchor import anchors_generation, sliding_anchors_all, pos_neg_iou
from heuristic_sampling import anchor_targets_bbox
from voc_data import voc_final
from loss import Loss
from PIL import Image
import numpy as np
import cv2
import time


def rpn(num_anchors):
    input_tensor = Input(shape=(224, 224, 3))
    # relu的全连接层
    x = Convolution2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        input_tensor)
    # rpn_cls与rpn_regression的分支
    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                            name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                           name='rpn_out_regress')(x)
    # summary
    model = Model(inputs=input_tensor, outputs=[x_class, x_regr], name='cls_regr_rpn')
    model.summary()
    return model


def train(num_anchors, imgs, labels_batch, regression_batch):
    model = rpn(num_anchors)
    adam = Adam(lr=0.001)
    cls_loss, regr_loss = Loss()
    model.compile(optimizer=adam, loss=[cls_loss, regr_loss], metrics=['accuracy'], loss_weights=[1, 1],
                  sample_weight_mode=None, weighted_metrics=None,
                  target_tensors=None)
    print("[INFO]网络RPN开始训练........")
    # 启发式采样中，标注为1的样本用于回归丨标注为0、1的样本用于分类
    # cls_inds =
    # regr_inds =
    history = model.fit(imgs, [labels_batch[:, :, 1][:, :, np.newaxis][:, inds, :], regression_batch[:, :, :4][:, inds, :]])


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
    # 生成所有映射回原图的anchors并进行启发式采样
    anchors = anchors_generation()
    all_anchors = sliding_anchors_all([width, height], [1, 1], anchors)
    # 计算得到分类、回归的目标
    labels_batch, regression_batch, num_anchors, inds = anchor_targets_bbox(all_anchors, all_images, all_annotations,
                                                                      len(classes_count), pos_overlap, neg_overlap,
                                                                      class_mapping)
    # 读取图片的rpn网络测试输入（图片的numpy、分类目标的numpy、回归的numpy）
    img_input = np.array(all_images)  # (1, 224, 224, 3)
    labels_input = labels_batch
    print(labels_input[:, :, 1][:, :, np.newaxis].shape)  # (1, 224*224*9, 1)
    regression_input = regression_batch
    print(regression_input[:, :, :4].shape)  # (1, 224*224*9, 4)
    # training
    start_time = time.time()
    train(num_anchors, img_input, labels_input, regression_input)
    end_time = time.time()
    print("时间消耗：{}".format(end_time - start_time))
