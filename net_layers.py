# -*- coding: utf-8 -*-
"""
Created on 2018/10/25 14:29

@author: royce.mao

resnet50的基础网络、共享feature map的RPN分支、共享feature map的ROI Pooling 分支
"""
from keras import layers
import keras.backend as K
from keras.layers import Input, Add,  Activation,  Convolution2D, AveragePooling2D, TimeDistributed
from FixedBatchNormalization import FixedBatchNormalization
from keras_applications.resnet50 import identity_block, conv_block
import tensorflow as tf

#======================================
# 一阶段
#======================================

def resnet50():
    """
    resnet50的基础特征提取网络
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
    return input_tensor, x

def rpn_layer(base_layer, num_anchors):
    """
    rpn 的网络结构
    :param num_anchors: 
    :return: 
    """
    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layer)
    # rpn_cls与rpn_regression的分支
    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                            name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                           name='rpn_out_regress')(x)
    return [x_class, x_regr]

#======================================
# 二阶段
#======================================

def roi_pooling_conv(img, rois, pooling_size, num_rois):
    """
    roi pooling conv 的逻辑
    :param img: 进入该层的feature map
    :param rois: feature map对应的rois
    :param pooling_size: 池化的固定尺寸
    :param num_rois: 
    :return: feature map与坐标映射后的rois对应区域经过resize之后的特征映射结果
    """
    input_shape = K.shape(img) # feature map尺寸大小
    channels = input_shape[0][3] # channels不变
    outputs = []
    for num in range(num_rois):
        # 一个batch一张图片
        x = rois[0, num, 0]
        y = rois[0, num, 1]
        w = rois[0, num, 2]
        h = rois[0, num, 3]
        # 改变张量的数据类型 int32
        x = K.cast(x, 'int32')
        y = K.cast(y, 'int32')
        w = K.cast(w, 'int32')
        h = K.cast(h, 'int32')
        # RoI区域映射feature map特征的resize，（w * h）->（pooling_size, pooling_size）
        ##不在（w * h）上做（pooling_size, pooling_size）的区域划分，然后在每个区域上做max_pooling？还是resize就是这个过程？
        rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (pooling_size, pooling_size))
        outputs.append(rs)
    final_output = K.concatenate(outputs, axis=0)
    final_output = K.reshape(final_output, (1, num_rois, pooling_size, pooling_size, channels)) # channels reshape为单独的维度
    final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4)) # 按照给定的模式重排一个张量的轴
    return final_output

def roi_pooling_layer(base_layer, rois, pooling_size, num_rois):
    """
    roi pooling layer 的网络结构
    :param base_layer: 
    :return: 
    """
    out_roi_pool = roi_pooling_conv(pooling_size, num_rois)([base_layer, rois])
    return out_roi_pool

def identity_block_td(input_tensor, block, kernel_size=3, filters=None, stage=5, trainable=True):
    """
    在identity_block基础上，引入了TimeDistributed封装，以产生针对各个时间步信号的独立RoIs处理
    :param input_tensor: 
    :param kernel_size: 
    :param filters: 
    :param stage: 
    :param block: 
    :param trainable: 
    :return: 
    """
    if filters is None:
        filters = [512, 512, 2048]
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block_td(input_tensor, block, input_shape, kernel_size=3, filters=None, stage=5, strides=(2, 2), trainable=True):
    """
    同样，在conv_block基础上，引入了TimeDistributed封装，以产生针对各个时间步信号的独立RoIs处理
    :param input_tensor: 
    :param kernel_size: 
    :param filters: 
    :param stage: 
    :param block: 
    :param input_shape: 
    :param strides: 
    :param trainable: 
    :return: 
    """
    if filters is None:
        filters = [512, 512, 2048]
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def fast_rcnn_layer(base_layer, input_shape, trainable):
    """
    stage_2 第2阶段fastrcnn的部分网络结构
    :param base_layer: 
    :param input_shape: 
    :return: 
    """
    x = conv_block_td(base_layer, block='a', input_shape=input_shape, trainable=trainable)
    x = identity_block_td(x, block='b', trainable=trainable)
    x = identity_block_td(x, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)
    return x


