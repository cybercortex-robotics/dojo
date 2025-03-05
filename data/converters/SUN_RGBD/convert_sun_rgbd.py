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


classes = [{'name': 'unknown', 'countable': False},
           {'name': 'wall', 'countable': False},
           {'name': 'floor', 'countable': False},
           {'name': 'cabinet', 'countable': True},
           {'name': 'bed', 'countable': True},
           {'name': 'chair', 'countable': True},
           {'name': 'sofa', 'countable': True},
           {'name': 'table', 'countable': True},
           {'name': 'door', 'countable': True},
           {'name': 'window', 'countable': True},
           {'name': 'bookshelf', 'countable': True},
           {'name': 'picture', 'countable': True},
           {'name': 'counter', 'countable': True},
           {'name': 'blinds', 'countable': False},
           {'name': 'desk', 'countable': True},
           {'name': 'shelves', 'countable': True},
           {'name': 'curtain', 'countable': False},
           {'name': 'dresser', 'countable': True},
           {'name': 'pillow', 'countable': True},
           {'name': 'mirror', 'countable': True},
           {'name': 'floor_mat', 'countable': False},
           {'name': 'clothes', 'countable': False},
           {'name': 'ceiling', 'countable': False},
           {'name': 'books', 'countable': True},
           {'name': 'fridge', 'countable': True},
           {'name': 'tv', 'countable': True},
           {'name': 'paper', 'countable': True},
           {'name': 'towel', 'countable': True},
           {'name': 'shower_curt', 'countable': True},
           {'name': 'box', 'countable': True},
           {'name': 'whiteboard', 'countable': True},
           {'name': 'person', 'countable': True},
           {'name': 'night_stand', 'countable': True},
           {'name': 'toilet', 'countable': True},
           {'name': 'sink', 'countable': True},
           {'name': 'lamp', 'countable': True},
           {'name': 'bathtub', 'countable': False},
           {'name': 'bag', 'countable': True}]


def convert_data(args):
    # Count data
    semantic_path = os.path.join(args.path, 'labels', 'train' if args.training == 'training' else 'test')

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
    # db.show_packing_info()
    db.create_db()

    # Prepare timestamp
    ts_start = int(time.time())
    ts_stop = ts_start
    sampling_time = 5

    # Add obj cls file
    obj_cls = create_obj_cls_file(classes=classes)
    db.add_custom(filter_id=2, data=obj_cls, name='object_classes.conf')

    # Receive data
    count_data = 0
    add_index = 1 if args.training == 'testing' else 5051

    with open(os.path.join(args.path, 'sunrgbd_{}_images.txt'.format(args.training)), 'r') as f:
        file = f.readlines()

        for path_idx, path in enumerate(file):
            image_name = path.split('/')[-1].split('.')[0]

            rgb_image_path = os.path.join(args.path, path[:-1])
            depth_image_path = os.path.join(args.path,
                                            os.path.join('/'.join(path.split('/')[:-2]), 'depth', image_name + '.png'))
            semseg_img_path = os.path.join(semantic_path, 'img-{:06}.png'.format(add_index + path_idx))

            # Check paths
            if not os.path.isfile(rgb_image_path):
                continue
            if not os.path.isfile(depth_image_path):
                continue
            if not os.path.isfile(semseg_img_path):
                continue

            semantic_img_raw = cv2.imread(semseg_img_path)

            # Pack data
            data = {
                'datastream_1': {  # image
                    'name': '{}.jpg'.format(ts_stop),
                    'image': rgb_image_path,
                    'image_right': depth_image_path
                },
                'datastream_2': {  # sem_seg
                    'name': '{}.png'.format(ts_stop),
                    'semantic': semantic_img_raw,
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

    print(' # Finished converting all data.')


def view_data(args):
    # Count data

    # View data
    for _ in _:
        pass
    print(' # Finished viewing all data.')


def main():
    parser = argparse.ArgumentParser(description='Convert SUN-RGBD Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default=r'C:\dev\sunrgbd-meta-data',
                        help='Path to the sunrgbd dataset')
    parser.add_argument('--training', '-t', default='testing', help='train / test')
    parser.add_argument('--output_path', '-o', default=r'C:\Databases\SUN_RGBD_test',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--width', default=640, type=int,
                        help='New width of the output image. Set to -1 to keep original size.')
    parser.add_argument('--height', default=480, type=int,
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