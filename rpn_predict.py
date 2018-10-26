# -*- coding: utf-8 -*-
"""
Created on 2018/10/26 16:13

@author: royce.mao

resnet50_rpn网络预测的cls得分（score）、regr目标的位置修正（revise）以生成原图的RoI。
"""

def rpn_predict_score():
    """
    proposals预测的前景评分
    :return: 
    """
    return 0


def rpn_predict_revise(proposals, dx1, dy1, dx2, dy2):
    """
    1阶段bbox_transform函数定义的回归目标在4个坐标(dx1,dy1,dx2,dy2)基础上，做位置修正
    :return: 
    """
    x1_target = dx1 * (proposals[:,2] - proposals[:,0]) + proposals[:, 0]
    y1_target = dy1 * (proposals[:,3] - proposals[:,1]) + proposals[:, 1]
    x2_target = dx2 * (proposals[:,2] - proposals[:,0]) + proposals[:, 2]
    y2_target = dy2 * (proposals[:,3] - proposals[:,1]) + proposals[:, 3]
    return x1_target, y1_target, x2_target, y2_target

