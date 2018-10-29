# -*- coding: utf-8 -*-
"""
Created on 2018/10/29 10:38

@author: royce.mao

Faster_rcnn 第2阶段，根据share的feature map以及rpn生成并映射的RoIs，组织具体类别的分类，和排除背景类的回归。
"""

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import TimeDistributed, Dense, Flatten
from frcnn_loss import class_loss_cls, class_loss_regr
from net_layers import resnet50, roi_pooling_layer, fast_rcnn_layer
import numpy as np


def res_roi_frcnn(rois_map, cls_target, regr_target, nb_classes = 21, trainable=False):
    """
    第2阶段，resnet50的基础特征提取网络 + roi pooling conv结构(特征映射) + cls_layer、regr_layer
    :param rois_map: rpn网络生成的feature map对应的rois
    :param cls_target: 分类目标
    :param regr_target: 回归目标
    :param num_rois: 一个batch图片对应的rois数量
    :return: 
    """
    input_shape = (len(rois_map),14,14,1024)
    # resnet50的16倍下采样的feature map
    base_layer = resnet50()
    # 结合feature map与feature map上对应映射的RoIs，做roi_pooling
    out_roi_pool = roi_pooling_layer(base_layer, rois_map, pooling_size=14, num_rois=len(rois_map))
    # 输出的是（None, num_rois, 2048)的子feature maps
    out_fast_rcnn = fast_rcnn_layer(out_roi_pool, input_shape=input_shape, trainable=True)
    # 因为是对num_rois个子feature maps分别处理的，这里使用timedistributed进行封装
    out = TimeDistributed(Flatten())(out_fast_rcnn)
    # 最终分支，同样使用封装器TimeDistributed来封装Dense，以产生针对各个时间步信号的独立全连接
    out_cls = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    ## （nb_classes-1）忽略了‘bg’背景类
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_cls, out_regr]

def gen_data_frcnn(rois_map, all_images, all_annotations, batch_size = 1):
    """
    生成器（迭代器），用于fit_generator边训练边生成训练数据
    :param all_anchors: 
    :param all_images: 
    :param all_annotations: 
    :param batch_size: 
    :return: 
    """
    length = len(all_images)
    for i in np.random.randint(0, length, size=batch_size):
        # 特征量：img、rois两个输入
        x = [all_images[i][np.newaxis,:,:], rois_map]
        # 标签量：cls、regr两个输出
        from roi_pooling import cls_target, regr_target
        rois, cls_target, pos_index, max_index = cls_target(rois_map, all_annotations, classifier_min_overlap=0.3, classifier_max_overlap=0.5)
        revise, regr_target = regr_target(rois_map, all_annotations, pos_index, max_index)
        y = [cls_target, regr_target]
        yield x, y



def train(input_tensor, rois_map, out_cls, out_regr, nb_classes):
    """
    训练过程
    :param input_tensor: 
    :param rois_map: 
    :param out_cls: 
    :param out_regr: 
    :return: 
    """
    model_fastrcnn = Model([input_tensor, rois_map], [out_cls, out_regr])
    adam = Adam(lr=1e-5)
    model_fastrcnn.compile(optimizer=adam,
                           loss=[class_loss_cls(), class_loss_regr(len(nb_classes) - 1)],
                           metrics={'dense_class_{}'.format(len(nb_classes)): 'accuracy'})
    print("[INFO]二阶段网络Fast_rcnn开始训练........")
    history = model_fastrcnn.fit_generator(generator = gen_data_frcnn(rois_map, all_images, all_annotations, batch_size = 1),
                                  steps_per_epoch = 1,
                                  epochs = 3)

if __name__ == "__main__":
    nb_classes = 21
    all_images = 0
    all_annotations = 0


