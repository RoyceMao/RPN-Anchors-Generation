# -*- coding: utf-8 -*-
"""
Created on 2018/10/29 10:38

@author: royce.mao

Faster_rcnn 第2阶段，根据share的feature map以及rpn生成并映射的RoIs，组织具体类别的分类，和排除背景类的回归。
"""
from __future__ import division
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import TimeDistributed, Dense, Flatten, Input
from nms import nms
from voc_data import voc_final
from rpn_train import regr_revise, resnet50_rpn, predict, regr_revise
from RoiPoolingConv import RoiPoolingConv
from frcnn_loss import class_loss_cls, class_loss_regr
from roi_pooling import cls_target, regr_target, proposal_to_roi
from anchor import anchors_generation, sliding_anchors_all, pos_neg_iou
from net_layers import resnet50, roi_pooling_layer, fast_rcnn_layer
from keras.utils import plot_model
import numpy as np
import time

def res_roi_frcnn(max_boxes, pooling_size, nb_classes):
    """
    第2阶段，resnet50的基础特征提取网络 + roi pooling conv结构(特征映射) + cls_layer、regr_layer
    :param rois_map: rpn网络生成的feature map对应的rois
    :param cls_target: 分类目标
    :param regr_target: 回归目标
    :param num_rois: 一个batch图片对应的rois数量
    :return: 
    """
    # input_tensor包括两个方面：1、feature map；2、rois
    input_rois = Input(shape=(max_boxes, 4)) # rois input
    input_shape = (max_boxes, 14, 14, 1024)
    # resnet50的16倍下采样的feature map
    input_tensor, base_layer = resnet50() # feature_map input（size是原图size，经过resnet50基础网络，成为feature_map size）
    # 结合feature map与feature map上对应映射的RoIs，做roi_pooling
    out_roi_pool = RoiPoolingConv(pooling_size, max_boxes)([base_layer, input_rois])
    ## 输出的是（None, num_rois, 14, 14, 1024)的子feature maps
    out_fast_rcnn = fast_rcnn_layer(out_roi_pool, input_shape=input_shape, trainable=True)
    ## 输出的是（None, num_rois, 1, 1, 2048)的子feature map
    # 因为是对num_rois个子feature maps分别处理的，这里使用timedistributed进行封装
    out = TimeDistributed(Flatten())(out_fast_rcnn)
    # 最终分支，同样使用封装器TimeDistributed来封装Dense，以产生针对各个时间步信号的独立全连接
    out_cls = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                              name='dense_class_{}'.format(nb_classes))(out)
    ## （nb_classes-1）忽略了‘bg’背景类
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    model = Model(inputs=[input_tensor, input_rois], outputs=[out_cls, out_regr], name='cls_regr_layer')
    # model.summary()
    return model


def gen_data_frcnn(model_rpn, all_images, all_annotations, batch_size=1):
    """
    生成器（迭代器），用于fit_generator边训练边生成训练数据
    :param all_anchors: 
    :param all_images: 
    :param all_annotations: 
    :param batch_size: 
    :return: 
    """
    # 一个一个batch生成训练所需的参数
    length = len(all_images)
    while True:
        for i in np.random.randint(0, length, size=batch_size):
            # model预测
            predict_imgs = predict(model_rpn, np.array(all_images[i][np.newaxis, :, :]))
            dx = predict_imgs[1].reshape(1, 1764, 4)[:, :, 0] # 1764=14*14*9
            dy = predict_imgs[1].reshape(1, 1764, 4)[:, :, 1]
            dw = predict_imgs[1].reshape(1, 1764, 4)[:, :, 2]
            dh = predict_imgs[1].reshape(1, 1764, 4)[:, :, 3]
            all_proposals = regr_revise(all_anchors, dx, dy, dw, dh)
            # 生成proposals
            proposals, probs = nms(np.column_stack((all_proposals, predict_imgs[0].ravel())), thresh=0.9,
                                   max_boxes=max_boxes)
            # print(type(np.array((all_annotations[i]))[0]))
            rois_pic, cls, pos_index, max_index = cls_target(proposals, np.array((all_annotations[i])),
                                                             classifier_min_overlap=0.1, classifier_max_overlap=0.5)
            shift = regr_target(rois_pic, np.array((all_annotations[i])), pos_index, max_index)
            rois_map = proposal_to_roi(rois_pic, stride)
            # 特征量：img、rois两个输入
            x = [all_images[i][np.newaxis, :, :], rois_map[np.newaxis, :, :]] # [1, num_rois, 4]
            # 标签量：cls、regr两个输出
            # y = [np.array((cls))[np.newaxis, :][:, :, np.newaxis], revise[np.newaxis, :, :]] # [1, num_rois, 20]、[1, num_rois, 80]
            # 开始将y转换为用于计算loss的y_true标签量
            ## y1分类目标（第3维列数等于nb_classe）
            y1 = np.zeros((max_boxes, nb_classes))
            for i in range(max_boxes):
                '''
                if cls[i] == 0.0:
                    y1[i] = [0, 0, 1]
                '''
                if cls[i] != 0.0:
                    a = np.zeros(nb_classes)
                    a[class_mapping[cls[i]]] = 1
                    y1[i] = a
            ## y2回归目标（第3维前4*(nb_classes-1)列是y1类别的repeat，用于标定正样本，后4*(nb_classes-1)列是对应正样本的真实回归目标）
            y2_true = np.zeros((max_boxes, 4*(nb_classes-1)))
            j = 0
            for i in range(max_boxes):
                if cls[i] in class_mapping.keys() and cls[i] != 'bg':
                    a = np.zeros(4*(nb_classes-1))
                    a[4*class_mapping[cls[i]] : 4*class_mapping[cls[i]]+4] = shift[j]
                    y2_true[i] = a
                    j += 1
            y2 = np.concatenate([np.repeat(y1[np.newaxis, :, :][:, :, :(nb_classes-1)], 4, axis=2), y2_true[np.newaxis, :, :]], axis=2)
            y = [y1[np.newaxis, :, :], y2]
            yield x, y


def train(model_rpn, max_boxes, pooling_size, nb_classes):
    """
    训练过程
    :param input_tensor: 
    :param rois_map: 
    :param out_cls: 
    :param out_regr: 
    :param nb_classes: 
    :return: 
    """
    # 【必须在train函数里面predict调用一下，不然generator函数里面直接调用会报错？？？】
    predict(model_rpn, np.array(all_images[0][np.newaxis, :, :]))
    model_fastrcnn = res_roi_frcnn(max_boxes, pooling_size, nb_classes)
    plot_model(model_fastrcnn, to_file='F:\\VOC2007\\model_fast_rcnn.png')
    try:
        print('loading weights from {}'.format('resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
        model_fastrcnn.load_weights('F:\\VOC2007\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
        print('加载预训练权重成功！')
    except:
        print('加载预训练权重失败！')
    adam = Adam(lr=1e-5)
    model_fastrcnn.compile(optimizer=adam,
                           loss=[class_loss_cls, class_loss_regr(nb_classes - 1)],
                           metrics={'dense_class_{}'.format(nb_classes): 'accuracy'})
    print("[INFO]二阶段网络Fast_rcnn开始训练........")
    history = model_fastrcnn.fit_generator(
        generator=gen_data_frcnn(model_rpn, all_images, all_annotations, batch_size=1),
        steps_per_epoch=1,
        epochs=25)
    return model_fastrcnn


if __name__ == "__main__":
    # 新增参数
    nb_classes = None # 总的类别数量
    max_boxes = 7 # 单张图片nms界定的rois数量【超过7就报错？？？】
    pooling_size = 14 # pooling的size
    # 准备voc的GT标注数据集
    data_path = "F:\\VOC2007"
    width = 14
    height = 14
    stride = [16, 16]
    class_mapping, classes_count, all_images, all_annotations = voc_final(data_path)
    nb_classes = len(class_mapping) + 1 # 类别数赋值，加上‘bg’类
    class_mapping['bg'] = len(class_mapping) # class_mapping字典新增‘bg’类
    # 生成所有映射回原图的anchors
    anchors = anchors_generation()
    all_anchors = sliding_anchors_all([width, height], stride, anchors)
    # 加载已训练的rpn模型权重
    model_rpn = resnet50_rpn(9)
    model_rpn.load_weights('F:\\VOC2007\\rpn.hdf5')
    print('RPN模型加载完毕！（用于训练中生成Proposals）')
    # 开始边生成数据边训练
    start_time = time.time()
    model_fast_rcnn = train(model_rpn, max_boxes, pooling_size, nb_classes)
    # model_fast_rcnn.save_weights('F:\\VOC2007\\fast_rcnn.hdf5')
    end_time = time.time()
    print("时间消耗：{}秒".format(end_time - start_time))
