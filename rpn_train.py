# -*- coding: utf-8 -*-
"""
Created on 2018/10/22 14:00

@author: royce.mao

# np.concatenate()方法不改变拼接后的numpy数组维度
# np.stack()方法一般来说会增加拼接后的numpy数组维度
# 这里训练rpn,用rpn生成proposals，并映射为RoIs之后，训练第2阶段的检测网络。
"""
from net_layers import resnet50, rpn_layer
from keras.optimizers import Adam
from keras.models import Model
from anchor import anchors_generation, sliding_anchors_all, pos_neg_iou
from heuristic_sampling import anchor_targets_bbox
from voc_data import voc_final
from rpn_loss import rpn_loss_cls, rpn_loss_regr
from roi_pooling import cls_target, regr_target, proposal_to_roi
from nms import nms
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import numpy as np
import time


def resnet50_rpn(num_anchors):
    """
    resnet50的基础特征提取网络+RPN的区域生成网络
    :param num_anchors: 
    :return: 
    """
    input_tensor, base_layer = resnet50()
    output_layer = rpn_layer(base_layer, num_anchors)
    model = Model(inputs=input_tensor, outputs=output_layer, name='cls_regr_rpn')
    # model.summary()
    return model


def gen_data_rpn(all_anchors,
                 all_images,
                 all_annotations,
                 batch_size=1):
    """
    生成器（迭代器），用于fit_generator边训练边生成训练数据
    :param all_anchors: 
    :param all_images: 
    :param all_annotations: 
    :param batch_size: 
    :return: 
    """
    length = len(all_images)
    while True:
         for i in np.random.randint(0, length, size=batch_size):
            x = all_images[i][np.newaxis, :, :]
            labels_batch, regression_batch, op_inds, inds = anchor_targets_bbox(all_anchors,
                                                                                    all_images[i][np.newaxis, :, :],
                                                                                    [all_annotations[i]],
                                                                                    len(classes_count), pos_overlap,
                                                                                    neg_overlap,
                                                                                    class_mapping)
            # print(labels_batch[:, :, 1][:, :, np.newaxis].shape)
            # 原始计算loss的y标签
            y1 = labels_batch[:, :, len(classes_count)][:, :, np.newaxis].reshape((1,14,14,9))
            y2 = regression_batch[:, :, :4].reshape(1,14,14,36)
            # y1_tmp中前景、背景类统一标识为1，忽略类标识为0，用以区分样本，只拿前景、背景类用于计算cls_loss
            y1_tmp = labels_batch[:, :, len(classes_count)][:, :, np.newaxis]
            y1_tmp[:, inds, :] = 1
            y1_tmp[:, op_inds, :] = 0
            # y_rpn_cls第4维前9列代表区分忽略类和非忽略类的numpy，y_rpn_regr第4维前36列代表区分正样本和非正样本的numpy
            y_rpn_cls = np.concatenate([y1_tmp.reshape((1,14,14,9)), y1], axis=3)
            y_rpn_regr = np.concatenate([np.repeat(y1, 4, axis=3), y2], axis=3)
            y = [y_rpn_cls, y_rpn_regr]
            yield x, y


def train(num_anchors):
    """
    训练过程
    :param num_anchors: 一个batch图片的anchors数量
    :return: 
    """
    model_rpn = resnet50_rpn(num_anchors)
    plot_model(model_rpn, to_file='F:\\VOC2007\\model_rpn.png')
    try:
        print('loading weights from {}'.format('resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
        model_rpn.load_weights('F:\\VOC2007\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
        print('加载预训练权重成功！')
    except:
        print('加载预训练权重失败！')
    adam = Adam(lr=1e-5) # 太高容易过拟合（影响后续nms），太低需要多个epochs收敛
    # callback =TensorBoard(log_dir='F:\\VOC2007\\logs', histogram_freq=0)
    model_rpn.compile(optimizer=adam, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)], metrics=['accuracy'], loss_weights=[1, 1],
                      sample_weight_mode=None, weighted_metrics=None,
                      target_tensors=None)
    print("[INFO]一阶段网络RPN开始训练........")
    # 启发式采样中，标注为1的样本用于回归丨标注为0、1的样本用于分类
    # history = model.fit_generator(imgs, [labels_batch[:, :, 1][:, :, np.newaxis][:, inds, :], regression_batch[:, :, :4][:, inds, :]])
    history = model_rpn.fit_generator(generator=gen_data_rpn(all_anchors, all_images, all_annotations, batch_size=1),
                                      # callbacks=callback,
                                      steps_per_epoch=1,
                                      epochs=25)
    return model_rpn

def predict(model_rpn, all_imgs):
    """
    预测过程
    :param model_rpn: 训练保存的模型（结构+权重）
    :param all_imgs: 读取的多个batch的图片[batch, 224, 224 ,3]
    :return: [(batch, 14, 14, 9),(batch, 14, 14, 36)]
    """
    predict_imgs = model_rpn.predict(all_imgs)
    return predict_imgs

def regr_revise(anchors, dx, dy, dw, dh):
    """
    第1阶段bbox_transform函数定义的回归目标在4个偏移量(dx,dy,dw,dh)基础上，做位置修正
    :return: 
    """
    x_target_center = dx * (anchors[:,2] - anchors[:,0]) + (anchors[:, 2] + anchors[:, 0]) / 2.0
    y_target_center = dy * (anchors[:,3] - anchors[:,1]) + (anchors[:, 3] + anchors[:, 1]) / 2.0
    w_target = np.exp(dw) * (anchors[:,2] - anchors[:,0])
    h_target = np.exp(dh) * (anchors[:,3] - anchors[:,1])
    x1_target = x_target_center - w_target / 2.0
    y1_target = y_target_center - h_target / 2.0
    x2_target = x_target_center + w_target / 2.0
    y2_target = y_target_center + h_target / 2.0
    return np.stack((x1_target.ravel(), y1_target.ravel(), x2_target.ravel(), y2_target.ravel())).T


if __name__ == "__main__":
    # 准备voc的GT标注数据集
    data_path = "F:\\VOC2007"
    width = 14
    height = 14
    stride = [16, 16]
    class_mapping, classes_count, all_images, all_annotations = voc_final(data_path)
    # 界定正、负样本的阈值边界
    pos_overlap = 0.5
    neg_overlap = 0.4
    # 生成所有映射回原图的anchors并进行启发式采样
    anchors = anchors_generation()
    all_anchors = sliding_anchors_all([width, height], stride, anchors)
    # print(all_anchors.shape)
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
    model_rpn = train(9)
    # model_rpn.save_weights('F:\\VOC2007\\rpn.hdf5')
    end_time = time.time()
    print("时间消耗：{}秒".format(end_time - start_time))
    # predicting并生成proposals（暂时拿原训练图做预测）
    for i in range(len(all_images)):
        # 由于nms的计算只能同时在一张图上进行,所以一张一张预测、nms筛选
        predict_imgs = predict(model_rpn, np.array(all_images[i][np.newaxis, :, :]))
        ## print(predict_imgs[0].shape) # 预测的所有anchors的probs
        ## print(predict_imgs[1].shape) # 预测的所有anchors的回归目标
        # 对all_anchors进行位置回归修正，得到all_proposals
        dx = predict_imgs[1].reshape(1, 1764, 4)[:, :, 0] # 1764=14*14*9
        dy = predict_imgs[1].reshape(1, 1764, 4)[:, :, 1]
        dw = predict_imgs[1].reshape(1, 1764, 4)[:, :, 2]
        dh = predict_imgs[1].reshape(1, 1764, 4)[:, :, 3]
        all_proposals = regr_revise(all_anchors, dx, dy, dw, dh)
        # 在all_proposals基础上进行nms筛选，并生成batch图片对应的最终proposals
        proposals, probs = nms(np.column_stack((all_proposals, predict_imgs[0].ravel())), thresh=0.9, max_boxes=10)
        print('生成的Proposals：\n{}'.format(proposals))
        ## thresh设置过低，max_boxe设置过高，都会导致最后满足nms条件的bboxes没max_boxes那么多，出现重复bboxes、probs
        #===========================================================================================================
        # 第1阶段整理完毕！接上第2阶段的部分内容测试
        # ===========================================================================================================
        # 标定2阶段全局的正、负样本（cls_target）
        rois_pic, cls, pos_index, max_index = cls_target(proposals, np.array((all_annotations[i]),dtype=np.ndarray),
                                                                classifier_min_overlap=0.1, classifier_max_overlap=0.5)
        # 标定2阶段全局的回归目标（shift），回归修正坐标值（revise_target）
        revise, shift = regr_target(rois_pic, np.array((all_annotations[i]),dtype=np.ndarray), pos_index, max_index)
        print('分类目标：\n{}'.format(cls))
        print('回归目标偏移：\n{}'.format(shift))
        # roi_pic 映射到 roi_feature_map（rpn后先做的回归修正，这里再做的映射）
        rois_map = proposal_to_roi(rois_pic, stride)
        print('映射后的RoIs：\n{}'.format(rois_map))
