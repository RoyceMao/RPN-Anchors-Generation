# -*- coding: utf-8 -*-
"""
Created on 2018/10/22 14:00

@author: royce.mao

rpn网络针对proposals（1：3采样后）的binary前景背景评分，以及bbox regression回归。
"""
from keras import layers
from keras.layers import Input, Convolution2D
from keras_applications.resnet50 import identity_block, conv_block
from keras.models import Model
from keras.optimizers import Adam
from anchor import anchors_generation, sliding_anchors_all, pos_neg_iou
from heuristic_sampling import anchor_targets_bbox
from voc_data import voc_final
from rpn_loss import Loss
import numpy as np
import time


def resnet50_rpn(num_anchors):
    """
    resnet50的基础特征提取网络+RPN的区域生成网络
    :param num_anchors: 
    :return: 
    """
    bn_axis = 3
    input_tensor = Input(shape=(224, 224, 3))
    # resnet50基础网络部分
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    ''' 
    # 选定trainable的层，默认全部训练
    base_model = Model(inputs=img_input, outputs=x)
    for layer in base_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
    '''
    # rpn部分，接上relu的全连接层
    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        x)
    # rpn_cls与rpn_regression的分支
    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                            name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                           name='rpn_out_regress')(x)
    # summary
    model = Model(inputs=input_tensor, outputs=[x_class, x_regr], name='cls_regr_rpn')
    # model.summary()
    return model


def gen_data(all_anchors,
             all_images,
             all_annotations,
             batch_size=1):
    """
    生成器（迭代器），用于fit_generator边训练边生成训练数据
    :param image_infos:
    :param batch_size:
    :return:
    """
    length = len(all_images)
    for i in np.random.randint(0, length, size=batch_size):
        x =  all_images[i][np.newaxis,:,:]
        labels_batch, regression_batch, num_anchors, inds = anchor_targets_bbox(all_anchors, all_images[i][np.newaxis,:,:],
                                                                                all_annotations,
                                                                                len(classes_count), pos_overlap,
                                                                                neg_overlap,
                                                                                class_mapping)
        # print(labels_batch[:, :, 1][:, :, np.newaxis][:, inds, :].shape)
        # print(regression_batch[:, :, :4][:, inds, :].shape)
        y = [labels_batch[:, :, 1][:, :, np.newaxis][:, inds, :], regression_batch[:, :, :4][:, inds, :]]
        yield x, y


def train(num_anchors):
    """
    训练过程
    :param num_anchors: 
    :param imgs: 读取的单batch图片目标
    :param labels_batch: 计算的分类目标
    :param regression_batch: 计算的回归目标
    :return: 
    """
    model = resnet50_rpn(num_anchors)
    adam = Adam(lr=0.001)
    cls_loss, regr_loss = Loss()
    model.compile(optimizer=adam, loss=[cls_loss, regr_loss], metrics=['accuracy'], loss_weights=[1, 1],
                  sample_weight_mode=None, weighted_metrics=None,
                  target_tensors=None)
    print("[INFO]网络RPN开始训练........")
    # 启发式采样中，标注为1的样本用于回归丨标注为0、1的样本用于分类
    # cls_inds =
    # regr_inds =
    # history = model.fit_generator(imgs, [labels_batch[:, :, 1][:, :, np.newaxis][:, inds, :], regression_batch[:, :, :4][:, inds, :]])
    history = model.fit_generator(generator = gen_data(all_anchors, all_images, all_annotations, batch_size = 1),
                                  steps_per_epoch = 1,
                                  epochs = 2)



if __name__ == "__main__":
    # 准备voc的GT标注数据集
    data_path = "F:\\VOC2007"
    width = 14
    height = 14
    stride = [16,16]
    class_mapping, classes_count, all_images, all_annotations = voc_final(data_path)
    # 界定正、负样本的阈值边界
    pos_overlap = 0.5
    neg_overlap = 0.4
    # 生成所有映射回原图的anchors并进行启发式采样
    anchors = anchors_generation()
    all_anchors = sliding_anchors_all([width, height], stride, anchors)
    # print(all_anchors)
    # print(all_annotations)
    '''
    # 一次性计算得到分类、回归的目标
    labels_batch, regression_batch, num_anchors, inds = anchor_targets_bbox(all_anchors, all_images, all_annotations,
                                                                      len(classes_count), pos_overlap, neg_overlap,
                                                                      class_mapping)
    # 一次性读取图片的resnet50_rpn网络测试输入（图片的numpy、分类目标的numpy、回归的numpy）
    img_input = np.array(all_images)  # (1, 224, 224, 3)
    labels_input = labels_batch
    print(labels_input[:, :, 1][:, :, np.newaxis].shape)  # (1, 224*224*9, 1)
    regression_input = regression_batch
    print(regression_input[:, :, :4].shape)  # (1, 224*224*9, 4)
    '''
    # training
    start_time = time.time()
    train(9)
    end_time = time.time()
    print("时间消耗：{}".format(end_time - start_time))
