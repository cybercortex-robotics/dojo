"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

"""
Script for converting SUN-RGBD Dataset to CyberCortex.AI format

Dataset Info:
    https://rgbd.cs.princeton.edu/

Dataset link:
    https://rgbd.cs.princeton.edu/data/SUNRGBD.zip
"""

import argparse
import os
import time
import json
import cv2
import glob
import numpy as np

from data.converters.CyC_DatabaseFormat import *


classes = [{'name': '<UNK>', 'countable': False}, {'name': 'beam', 'countable': False},
           {'name': 'board', 'countable': True}, {'name': 'bookcase', 'countable': True},
           {'name': 'ceiling', 'countable': False}, {'name': 'chair', 'countable': True},
           {'name': 'clutter', 'countable': False}, {'name': 'column', 'countable': True},
           {'name': 'door', 'countable': True}, {'name': 'floor', 'countable': False},
           {'name': 'sofa', 'countable': True}, {'name': 'table', 'countable': True},
           {'name': 'wall', 'countable': False}, {'name': 'window', 'countable': True}]


CLASSES = {
    '<UNK>': 0,
    'beam': 1,
    'board': 2,
    'bookcase': 3,
    'ceiling': 4,
    'chair': 5,
    'clutter': 6,
    'column': 7,
    'door': 8,
    'floor': 9,
    'sofa': 10,
    'table': 11,
    'wall': 12,
    'window': 13
}


def get_index(in_image: np.ndarray = None) -> np.ndarray:
    semantic_img_idxs = np.zeros(shape=(in_image.shape[0], in_image.shape[1]))
    semantic_img_idxs[:, :] = in_image[:, :, 0] * 256 * 256 + in_image[:, :, 1] * 256 + in_image[:, :, 2]
    return semantic_img_idxs


def get_class(in_image: np.ndarray = None, json_data: list = None) -> np.ndarray:
    semantic_classes_img = np.zeros_like(a=in_image)
    for i in range(in_image.shape[0]):
        for j in range(in_image.shape[1]):
            assert in_image[i, j] < len(json_data)

            semantic_classes_img[i, j] = CLASSES[json_data[int(in_image[i, j])].split('_')[0]]
    return semantic_classes_img


def convert_data(args):
    # 3D_paths
    base_path_3d = os.path.join(r'/'.join(args.path.split('\\')[:-1]), '3d')
    info_3d_path = os.path.join(base_path_3d, 'pointcloud.mat')

    # Count data
    rgb_path = os.path.join(args.path, 'rgb')
    depth_path = os.path.join(args.path, 'depth')
    semantic_path = os.path.join(args.path, 'semantic')

    total_images = len(os.listdir(semantic_path))

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    # Create empty database with the index db_idx for every stream
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='image', filter_id=1, filter_name='CameraFront')
    db.add_stream(filter_type='semseg', filter_id=2, filter_name='SemSegFront', input_sources=[1])
    db.add_stream(filter_type='lidar', filter_id=3, filter_name='LidarFront')
    db.add_stream(filter_type='3D_obj_det', filter_id=4, filter_name='3DObjectDetection')
    # db.show_packing_info()
    db.create_db()

    # Prepare timestamp
    ts_start = int(time.time())
    ts_stop = ts_start
    sampling_time = 5

    # Add obj cls file
    obj_cls = create_obj_cls_file(classes=classes)
    db.add_custom(filter_id=2, data=obj_cls, name='object_classes.conf')

    # read json
    json_file = open(r'C:/Databases/2D-3D-Semantics/assets/semantic_labels.json')
    json_data = json.load(json_file)

    # Receive data
    count_data = 0

    for path in os.listdir(semantic_path):
        semantic_img_path = os.path.join(semantic_path, path)
        depth_img_path = os.path.join(depth_path, '_'.join(path.split('_')[:-1]) + '_depth.png')
        rgb_img_path = os.path.join(rgb_path, '_'.join(path.split('_')[:-1]) + '_rgb.png')

        # Check paths
        if not os.path.isfile(depth_img_path):
            continue
        if not os.path.isfile(rgb_img_path):
            continue

        semantic_img_raw = cv2.imread(semantic_img_path)
        semantic_img_idxs = get_index(in_image=semantic_img_raw)
        semantic_img_clss = get_class(in_image=semantic_img_idxs, json_data=json_data)

        # Pack data
        data = {
            'datastream_1': {  # image
                'name': '{}.jpg'.format(ts_stop),
                'image': '{}/{}.png'.format(rgb_path, '_'.join(path.split('_')[:-1]) + '_rgb'),
                'image_right': '{}/{}.png'.format(depth_path, '_'.join(path.split('_')[:-1]) + '_depth')
            },
            'datastream_2': {  # sem_seg
                'name': '{}.png'.format(ts_stop),
                'semantic': semantic_img_clss,
                'cls': None,
                'instance': None,
                'points': None
            }
        }

        # Add data to database
        db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

        # Advance timestamp
        ts_start = ts_stop
        ts_stop += sampling_time

        # Print progress
        count_data += 1
        if count_data % 10 == 0:
            print('Done {0} out of {1}.'.format(count_data, total_images))

        if count_data >= 99:
            break

    print(' # Finished converting all data.')


def view_data(args):
    # Count data

    # View data
    for _ in _:
        pass
    print(' # Finished viewing all data.')


def main():
    parser = argparse.ArgumentParser(description='Convert SUN-RGBD Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default=r'C:\Databases\area_1\data',
                        help='Path to the waymo dataset')
    parser.add_argument('--output_path', '-o', default=r'C:\Databases\Stanford2D3D_converted',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--width', default=800, type=int,
                        help='New width of the output image. Set to -1 to keep original size.')
    parser.add_argument('--height', default=800, type=int,
                        help='New height of the output image. Set to -1 to keep original size.')

    parser.add_argument('--mode', '-m', default='convert', help='view / convert')

    args = parser.parse_args()

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(parser.parse_args().mode))


if __name__ == '__main__':
    main()
