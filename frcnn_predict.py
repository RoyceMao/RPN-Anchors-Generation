# -*- coding: utf-8 -*-
"""
Created on 2018/11/2 13:51

@author: royce.mao

# 2阶段faster_rcnn交替训练完毕后的预测Predict。
"""
from __future__ import division
from nms import nms
from voc_data import voc_final
from frcnn_train import res_roi_frcnn
from rpn_train import regr_revise, resnet50_rpn, predict, regr_revise
from anchor import anchors_generation, sliding_anchors_all, pos_neg_iou
from roi_pooling import cls_target, regr_target, proposal_to_roi
import numpy as np


def frcnn_prediction(all_images, all_anchors, all_annotations, stride, class_mapping, model_rpn, model_fast_rcnn, max_boxes):
    # 为第2阶段的预测准备RoIs输入
    for i in range(len(all_images)):
        predict_rpn = predict(model_rpn, np.array(all_images[i][np.newaxis, :, :]))
        dx = predict_rpn[1].reshape(1, 1764, 4)[:, :, 0]  # 1764=14*14*9
        dy = predict_rpn[1].reshape(1, 1764, 4)[:, :, 1]
        dw = predict_rpn[1].reshape(1, 1764, 4)[:, :, 2]
        dh = predict_rpn[1].reshape(1, 1764, 4)[:, :, 3]
        all_proposals = regr_revise(all_anchors, dx, dy, dw, dh)
        proposals, probs = nms(np.column_stack((all_proposals, predict_rpn[0].ravel())), thresh=0.9,
                               max_boxes=max_boxes)
        pred_rois_pic, cls, pos_index, max_index = cls_target(proposals, np.array((all_annotations[i])),
                                                              classifier_min_overlap=0.1, classifier_max_overlap=0.5)
        pred_rois_map = proposal_to_roi(pred_rois_pic, stride)
        # 开始预测
        predict_imgs = model_fast_rcnn.predict(
            [np.array(all_images[i][np.newaxis, :, :]), pred_rois_map[np.newaxis, :, :]])
        # 位置修正
        ## softmax概率最大的类别及index
        softmax_index = [np.argmax(predict_imgs[0][0][i]) for i in range(max_boxes)]
        softmax_prob = [np.max(predict_imgs[0][0][i]) for i in range(max_boxes)]
        ## index对应的回归目标提取
        softmax_regr = np.array(
            [predict_imgs[1][0][i][4 * softmax_index[i]: 4 * softmax_index[i] + 4] for i in range(max_boxes)])
        d_x = softmax_regr[:, 0]
        d_y = softmax_regr[:, 1]
        d_w = softmax_regr[:, 2]
        d_h = softmax_regr[:, 3]
        ## 对应目标的位置修正
        final_boxes = regr_revise(pred_rois_pic, d_x, d_y, d_w, d_h)
        print('检测框类别预测：\n{}'.format(
            [list(class_mapping.keys())[list(class_mapping.values()).index(softmax_index[i])] for i in
             range(max_boxes)]))
        print('检测框最高类别概率预测：\n{}'.format(softmax_prob))
        print('检测框位置预测：\n{}'.format(final_boxes))
        print('实际的Ground Truth：\n{}'.format(all_annotations[0]))
    # 暂时先return一张图片的prediction结果
    return [list(class_mapping.keys())[list(class_mapping.values()).index(softmax_index[i])] for i in
             range(max_boxes)], softmax_prob, final_boxes

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
    # 加载已训练的fast_rcnn模型权重
    model_fast_rcnn = res_roi_frcnn(max_boxes, pooling_size, nb_classes)
    model_fast_rcnn.load_weights('F:\\VOC2007\\fast_rcnn.hdf5')
    print('Fast_rcnn模型加载完毕！（用于训练中生成Proposals）')
    # 预测
    frcnn_prediction(all_images, all_anchors, all_annotations, stride, class_mapping, model_rpn, model_fast_rcnn, max_boxes)