"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

# import os
# import glob
# from scipy.io import loadmat
#
# classes = ['person?', 'person', 'person-fa', 'people']
#
# frame_size = (640, 480)
# number_of_truth_boxes = 0
#
# path_to_dataset = r'C:/Datasets/caltech'
# path_to_labels = path_to_dataset + r'/labels'
# path_to_annotations = path_to_dataset + r'/annotations'
#
# if not os.path.exists(path_to_labels):
#     os.makedirs(path_to_labels)
#
#
# def convert_box_format_2_yolo(box):
#     (box_x, box_y, box_w, box_h) = box
#     x = box_x + box_w / 2.0
#     y = box_y + box_h / 2.0
#     w = box_w
#     h = box_h
#     return x, y, w, h
#
#
# for caltech_set in sorted(glob.glob(path_to_annotations + r'/set*')):
#     set_nr = int(os.path.basename(caltech_set).replace('set', ''))
#
#     for caltech_annotation in sorted(glob.glob(caltech_set + '/*.vbb')):
#         vbb = loadmat(caltech_annotation)
#         obj_lists = vbb['A'][0][0][1][0]
#         obj_label = [str(v[0]) for v in vbb['A'][0][0][4][0]]
#         video_id = os.path.splitext(os.path.basename(caltech_annotation))[0]
#
#         for frame_id, obj in enumerate(obj_lists):
#             if len(obj) > 0:
#                 labels = ''
#                 for pedestrian_id, pedestrian_position in zip(obj['id'][0], obj['pos'][0]):
#                     pedestrian_id = int(pedestrian_id[0][0]) - 1
#                     pedestrian_position = pedestrian_position[0].tolist()
#                     class_index = classes.index(obj_label[pedestrian_id])
#                     yolo_format = convert_box_format_2_yolo(pedestrian_position)
#                     x, y, w, h = yolo_format
#                     labels += str(set_nr) + "," + str(video_id) + "," + str(frame_id) + "," + str(class_index) + "," + \
#                               str(x) + "," + str(y) + "," + str(w) + "," + str(h) + "\n"
#                     number_of_truth_boxes += 1
#                 if not labels:
#                     continue
#
#                 with open(path_to_labels + '/labels.txt', 'a') as writer:
#                     writer.write(labels)
#                     writer.close()

"""
    Caltech Pedestrian Dataster converter.
"""

import os
import csv
from datetime import datetime
from shutil import copy
import numpy as np
import math
import cv2 as cv
import glob

from toolkit.object_classes import ObjectClasses
from data import types_CyC_TYPES
import global_config

CFG = global_config.cfg
object_classes = ObjectClasses(CFG.CyC_INFERENCE.OBJECT_CLASSES_PATH)

caltech_classes_mapper = {'person?': object_classes.object_classes[0][0],
                          'person': object_classes.object_classes[0][0],
                          'person-fa': object_classes.object_classes[0][0],
                          'people': object_classes.object_classes[0][0]
                          }

path_to_dataset = r'C:/Datasets/caltech/'
path_to_destination_folder = r'C:/Datasets/Caltech2CyC/'

SETS = ['set00/', 'set01/', 'set02/', 'set03/', 'set04/', 'set05/', 'set06/', 'set07/', 'set08/', 'set09/', 'set10/']
V_SEQ = ['V000.seq', 'V001.seq', 'V002.seq', 'V003.seq', 'V004.seq', 'V005.seq', 'V006.seq', 'V007.seq', 'V008.seq',
         'V009.seq', 'V010.seq', 'V011.seq', 'V012.seq', 'V013.seq', 'V014.seq', 'V015.seq', 'V016.seq', 'V017.seq',
         'V018.seq', 'V019.seq']


def makedir(dir_path, dir_name):
    if os.path.isdir(dir_path + dir_name):
        print("Folder " + dir_path + dir_name + " already exist. Skipp command.")
    else:
        try:
            os.makedirs(dir_path + dir_name)
        except OSError:
            pass


def make_sensor(path_to_destination_folder, set_folder, cam_sensors):
    global SENSOR_ITERATOR

    if not cam_sensors[SENSOR_ITERATOR]:
        makedir(path_to_destination_folder + set_folder + 'datastrem_1_' + str(SENSOR_ITERATOR + 1) + '/', 'samples/')
        SENSOR_ITERATOR += 1
    else:
        makedir(path_to_destination_folder + set_folder + 'datastrem_1_' + str(SENSOR_ITERATOR + 1) + '/', 'samples/')
        makedir(path_to_destination_folder + set_folder + 'datastrem_1_' + str(SENSOR_ITERATOR + 1) + '/samples/',
                'left')
        makedir(path_to_destination_folder + set_folder + 'datastrem_1_' + str(SENSOR_ITERATOR + 1) + '/samples/',
                'right')
        SENSOR_ITERATOR += 1


def map_caltech_2_CyC_format(path_to_destination_folder, set_folder, sensors):
    for sensor in sensors:
        if 'CAM' == sensor.split("_")[0]:
            for dname in sorted(glob.glob(path_to_dataset + 'set*')):
                for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
                    cap = cv.VideoCapture(fn)
                    i = 0
                    while True:
                        ret, frame = cap.read()
                        print(frame)


def generate_datablock_descriptor(SET, SENSORS):
    vision_core_id = 1

    CyC_types_path = CFG.CyC_INFERENCE.TYPES_FILE
    if not os.path.exists(CyC_types_path):
        print('CyC_TYPES.h: {0} not found'.format(CyC_types_path))
    CyC_FILTER_TYPE_ENUM_NAME = "CyC_FILTER_TYPE"
    CyC_DATA_TYPE_ENUM_NAME = "CyC_DATA_TYPE"

    filter_type = types_CyC_TYPES.get_datatypes_as_dict(CyC_types_path, CyC_FILTER_TYPE_ENUM_NAME)
    data_type = types_CyC_TYPES.get_datatypes_as_dict(CyC_types_path, CyC_DATA_TYPE_ENUM_NAME)

    SENSORS_FILTER = {
        'CAM': 'CyC_MONO_CAMERA_FILTER_TYPE',
        '2D': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE'
    }

    SENSORS_DATA = {
        'CAM': 'CyC_IMAGE',
        '2D': 'CyC_2D_ROIS'
    }

    with open(path_to_destination_folder + SET + 'datablock_descriptor.csv',
              mode='a',
              newline='') as blockchain_descriptor_file:
        blockchain_descriptor_writer = csv.writer(blockchain_descriptor_file, delimiter=',')
        header = list()
        header.append('vision_core_id')
        header.append('filter_id')
        header.append('name')
        header.append('type')
        header.append('output_data_type')
        header.append('input_sources')

        blockchain_descriptor_writer.writerow([column for column in header])

        last_cam_idx = 0

        for idx, sensor in enumerate(SENSORS):
            if 'CAM' == sensor.split("_")[0]:
                row = list()
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append(sensor)
                row.append(str(filter_type[SENSORS_FILTER['CAM']]))
                row.append(str(data_type[SENSORS_DATA['CAM']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                last_cam_idx = idx
            elif '2D' == sensor.split("_")[0]:
                row = list()
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append(sensor)
                row.append(str(filter_type[SENSORS_FILTER['2D']]))
                row.append(str(data_type[SENSORS_DATA['2D']]))
                row.append('{1-' + str(idx - last_cam_idx) + ';1-' + str(last_cam_idx + 1) + '}')
                blockchain_descriptor_writer.writerow(column for column in row)


def main():
    for SET in SETS:
        SENSORS = []
        CAM_SENSORS = []

        global SENSOR_ITERATOR
        SENSOR_ITERATOR = 0

        for seq in V_SEQ:
            if os.path.exists(path_to_dataset + SET + seq):
                SENSORS.append('CAM_FRONT_' + seq.split('.')[0])
                CAM_SENSORS.append(True)

        for seq in V_SEQ:
            if os.path.exists(path_to_dataset + SET + seq):
                SENSORS.append('2D_CAM_FRONT_' + seq.split('.')[0])
                CAM_SENSORS.append(False)

        makedir(path_to_destination_folder, SET)

        for i in range(len(SENSORS)):
            make_sensor(path_to_destination_folder, SET, CAM_SENSORS)

        map_caltech_2_CyC_format(path_to_destination_folder, SET, SENSORS)

        generate_datablock_descriptor(SET, SENSORS)


if __name__ == "__main__":
    main()
