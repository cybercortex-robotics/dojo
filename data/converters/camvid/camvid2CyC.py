"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import os
import shutil
import time

import cv2
import numpy as np

import sys
sys.path.insert(1, '../../')
from CyC_db_interface import CyC_FilterType, CyC_DataType

# Path to camera images folder
path_to_images = r'D:\pionner\camvid\raw'
# Path to segmented images folder
path_to_labels = r'D:\pionner\camvid\labeled'
# Path to new datastream folder
path_to_output_dataset = r'D:\pionner\camvid\db'

camera_filter_id = 1
segmentation_filter_id = 10

CyC_core_id = 1

csv_header = 'timestamp_start,timestamp_stop,sampling_time,left_file_path_0,right_file_path_0'
blockchain_header = 'vision_core_id,filter_id,name,type,output_data_type,input_sources'


def main():
    images_datastream_name = 'datastream_{}_{}'.format(CyC_core_id, camera_filter_id)
    labels_datastream_name = 'datastream_{}_{}'.format(CyC_core_id, segmentation_filter_id)
    images_folder = os.path.join(path_to_output_dataset, images_datastream_name, 'samples', 'left')
    labels_folder = os.path.join(path_to_output_dataset, labels_datastream_name, 'samples', 'left')

    try:
        os.makedirs(images_folder)
    except OSError:
        pass

    try:
        os.makedirs(labels_folder)
    except OSError:
        pass

    images_csv_file = open(os.path.join(path_to_output_dataset, images_datastream_name, 'data_descriptor.csv'), 'w')
    images_csv_file.write(csv_header + '\n')

    labels_csv_file = open(os.path.join(path_to_output_dataset, labels_datastream_name, 'data_descriptor.csv'), 'w')
    labels_csv_file.write(csv_header + '\n')

    images_list = os.listdir(path_to_images)
    labels_list = os.listdir(path_to_labels)

    timestamps_file = open(os.path.join(path_to_output_dataset, 'sampling_timestamps_sync.csv'), 'w')
    timestamps_header = '{},{},{}'.format('timestamp_stop', images_datastream_name, labels_datastream_name)
    timestamps_file.write(timestamps_header + '\n')

    dt = 1000  # 1Hz
    initial_timestamp = int(time.time() * 1000)

    ts_start = initial_timestamp
    ts_stop = initial_timestamp + dt
    for image, label in zip(images_list, labels_list):
        src = os.path.join(path_to_images, image)
        dst = os.path.join(images_folder, image)
        shutil.copyfile(src, dst)
        line = '{},{},{},{},'.format(ts_start, ts_stop, dt, os.path.join(r'samples\left', image))
        images_csv_file.write(line + '\n')

        src = os.path.join(path_to_labels, label)
        dst = os.path.join(labels_folder, label)

        label_image = cv2.imread(src)
        label_image[np.where((label_image == [64, 128, 192]).all(axis=2))] = [0,  64,  64]  ## Replace child class (192. 128. 64) RGB with pedestrian (64, 64, 0)
        cv2.imwrite(dst, label_image)

        line = '{},{},{},{},'.format(ts_start, ts_stop, dt, os.path.join(r'samples\left', label))
        labels_csv_file.write(line + '\n')

        timestamps_file.write('{},{},{}\n'.format(ts_stop, ts_stop, ts_stop))

        ts_start = ts_start + dt
        ts_stop = ts_stop + dt

    blockchain_file = open(os.path.join(path_to_output_dataset, 'datablock_descriptor.csv'), 'w')
    blockchain_file.write(blockchain_header + '\n')
    blockchain_file.write('{},{},{},{},{},\n'.format(CyC_core_id, camera_filter_id,
                                                     'CameraFront',
                                                     CyC_FilterType.CyC_MONO_CAMERA_FILTER_TYPE,
                                                     CyC_DataType.CyC_IMAGE))

    _inputs = '{{{0}-{1}}}'.format(CyC_core_id, camera_filter_id)
    blockchain_file.write('{},{},{},{},{},{}\n'.format(CyC_core_id, segmentation_filter_id,
                                                       "SemanticSegCamFront",
                                                       CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE,
                                                       CyC_DataType.CyC_IMAGE,
                                                       _inputs))


if __name__ == '__main__':
    main()
