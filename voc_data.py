# -*- coding: utf-8 -*-
"""
Created on 2018/10/19 09:09

@author: royce.mao

voc标注数据集的处理与解析
"""

import os
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

def get_voc_data(data_path):
    """
    获取VOC数据集的元数据信息：名称、类型、目标(类别、边框位置)
    :param data_path:
    :return:
    """
    all_imgs = []
    classes_count = {}
    class_mapping = {}
    visualise = False
    annot_path = os.path.join(data_path, 'Annotations')
    imgs_path = os.path.join(data_path, 'JPEGImages')
    imgsets_path_trainval = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
    imgsets_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')

    trainval_files = []
    test_files = []
    try:
        with open(imgsets_path_trainval) as f:
            for line in f:
                trainval_files.append(line.strip() + '.jpg')
    except Exception as e:
        print(e)

    try:
        with open(imgsets_path_test) as f:
            for line in f:
                test_files.append(line.strip() + '.jpg')
    except Exception as e:
        if data_path[-7:] == 'VOC2007':
            pass
        else:
            print(e)

    annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
    idx = 0
    for annot in annots:
        try:
            idx += 1
            et = ET.parse(annot)
            element = et.getroot()
            element_objs = element.findall('object')
            element_filename = element.find('filename').text
            element_width = int(element.find('size').find('width').text)
            element_height = int(element.find('size').find('height').text)

            if len(element_objs) > 0:
                annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
                                   'height': element_height, 'bboxes': []}
                if element_filename in trainval_files:
                    annotation_data['imageset'] = 'trainval'
                elif element_filename in test_files:
                    annotation_data['imageset'] = 'test'
                else:
                    annotation_data['imageset'] = 'trainval'

            for element_obj in element_objs:
                class_name = element_obj.find('name').text
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)

                obj_bbox = element_obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                difficulty = int(element_obj.find('difficult').text) == 1
                annotation_data['bboxes'].append(
                    {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
            all_imgs.append(annotation_data)
        except Exception as e:
            print(e)
            continue
    print("class_map:{}".format(class_mapping))
    return class_mapping, all_imgs, classes_count


def parse_data_detect(all_imgs_infos):
    """
    把之前生成的all_imgs字典格式数据，按既定缩放比例,解析为目标检测需要的结果
    (x1, x2, y1, y2, cls)
    :param image_infos:
    :return:
    """
    all_images = []
    all_annotations = {}
    for image_info in all_imgs_infos:
        img_path = image_info["filepath"]
        img_data = np.array(Image.open(img_path).convert("RGB"))
        img_data = cv2.resize(img_data, (224, 224), interpolation=cv2.INTER_CUBIC)
        width_info = image_info["width"]
        height_info = image_info["height"]
        if 'bboxes' in image_info.keys():
            list = []
            for box in image_info['bboxes']:
                cls = box['class']
                '''
                x1 = box['x1']
                x2 = box['x2']
                y1 = box['y1']
                y2 = box['y2']
                '''
                x1 = box['x1']*(224/width_info)
                x2 = box['x2']*(224/width_info)
                y1 = box['y1']*(224/height_info)
                y2 = box['y2']*(224/height_info)
                (x1, x2, y1, y2) = [round(x) for x in [x1, x2, y1, y2]]
                list.append([x1, x2, y1, y2, cls])
            all_annotations[image_info["filepath"].split("\\")[-1]] = list
        all_images.append(img_data)
    return all_images, all_annotations

def voc_final(data_path):
    class_mapping, all_imgs, classes_count = get_voc_data(data_path)
    print(all_imgs)
    all_images, all_annotations = parse_data_detect(all_imgs)
    return class_mapping, classes_count, all_images, all_annotations

if __name__ == "__main__":
    data_path = "F:\\VOC2007"
    width = 224
    height = 224
    class_mapping, classes_count, all_images, all_annotations = voc_final(data_path)
    print(list(all_images[0].shape[:2]))
    print(np.array(all_images).shape)
    for index, (image, annotations) in enumerate(zip(all_images, all_annotations.values())):
        print(annotations)