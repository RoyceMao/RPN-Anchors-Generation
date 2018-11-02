"""
this is common visualize utils to show boxes in detection or tracking,
this file support both cv2 or PIL library, with separately methods
"""
import cv2
from frcnn_predict import frcnn_prediction
from voc_data import voc_final
from frcnn_train import res_roi_frcnn
from rpn_train import regr_revise, resnet50_rpn, predict, regr_revise
from anchor import anchors_generation, sliding_anchors_all, pos_neg_iou
from PIL import Image
import numpy as np
import colorsys


def _create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).
    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.
    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).
    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]
    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def _create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id or class in detection (tag).
    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.
    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).
    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]
    """
    r, g, b = _create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


def draw_boxes_and_label_on_image(img, class_label_map, class_boxes_map):
    """
    this method using cv2 to show boxes on image with various class labels
    :param img:
    :param class_label_map: {1: 'Car', 2: 'Pedestrian'}
    :param class_boxes_map: {1: [box1, box2..], 2: [..]}, in every box is [bb_left, bb_top, bb_width, bb_height, prob]
    :return:
    """
    for c, boxes in class_boxes_map.items():
        for box in boxes:
            assert len(box) == 5, 'class_boxes_map every item must be [bb_left, bb_top, bb_width, bb_height, prob]'
            # checking box order is bb_left, bb_top, bb_width, bb_height
            # make sure all box should be int for OpenCV
            bb_left = int(box[0])
            bb_top = int(box[1])
            bb_width = int(box[2]-box[0])
            bb_height = int(box[3]-box[1])

            # prob will round 2 digits
            prob = round(box[4], 2)
            unique_color = _create_unique_color_uchar(c)
            cv2.rectangle(img, (bb_left, bb_top), (bb_width, bb_height), unique_color, 2)

            text_label = '{} {}'.format(class_label_map[c], prob)
            (ret_val, base_line) = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            text_org = (bb_left, bb_top - 0)

            cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line - 5),
                          (text_org[0] + ret_val[0] + 5, text_org[1] - ret_val[1] + 5), unique_color, 2)
            # this rectangle for fill text rect
            cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line - 5),
                          (text_org[0] + ret_val[0] + 5, text_org[1] - ret_val[1] + 5),
                          unique_color, -1)
            cv2.putText(img, text_label, text_org, cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    return img

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
    print('RPN模型加载完毕！')
    # 加载已训练的fast_rcnn模型权重
    model_fast_rcnn = res_roi_frcnn(max_boxes, pooling_size, nb_classes)
    model_fast_rcnn.load_weights('F:\\VOC2007\\fast_rcnn.hdf5')
    print('Fast_rcnn模型加载完毕！')
    # 预测
    cls_map, probs, regr_map = frcnn_prediction(all_images, all_anchors, all_annotations, stride, class_mapping, model_rpn, model_fast_rcnn, max_boxes)
    # ================================================================
    # 暂时先打印一张resize到224*224图片检测结果的可视化
    draw_imgs = draw_boxes_and_label_on_image(all_images[0], {1:cls_map}, {1:np.column_stack((regr_map, probs))})
    # img save
    im = Image.fromarray(draw_imgs)
    im.save('F:\\VOC2007\\test.png')
