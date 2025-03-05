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


CLASSES = [{'name': 'background', 'countable': False}, {'name': 'person', 'countable': True},
           {'name': 'cabinet', 'countable': True}]


def convert_data(args):
    # Count data
    rgb_path = os.path.join(args.path, 'rgb')
    depth_path = os.path.join(args.path, 'depth')

    total_images = len(os.listdir(rgb_path))

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    # Create empty database with the index db_idx for every stream
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='image', filter_id=1, filter_name='CameraFront')
    # db.show_packing_info()
    db.create_db()

    # Prepare timestamp
    ts_start = int(time.time())
    ts_stop = ts_start
    sampling_time = 5

    # Add obj cls file
    obj_cls = create_obj_cls_file(classes=CLASSES)
    db.add_custom(filter_id=1, data=obj_cls, name='object_classes.conf')

    # Receive data
    count_data = 0

    for path in os.listdir(rgb_path):
        # depth_img_path = os.path.join(depth_path, '_'.join(path.split('_')[:-1]) + '_depth.png')
        depth_img_path = os.path.join(depth_path, path)
        rgb_img_path = os.path.join(rgb_path, path)

        # Check paths
        if not os.path.isfile(depth_img_path):
            continue
        if not os.path.isfile(rgb_img_path):
            continue

        # Pack data
        data = {
            'datastream_1': {  # image
                'name': '{}.jpg'.format(ts_stop),
                'image': rgb_img_path,
                'image_right': depth_img_path
            }
        }

        # Add data to database
        db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

        # Advance timestamp
        ts_start = ts_stop
        ts_stop += sampling_time

        # Print progress
        count_data += 1

        print('Done {0} out of {1}.'.format(count_data, total_images))

        # if count_data >= 99:
        #     break

    print(' # Finished converting all data.')


def view_data(args):
    # Count data

    # View data
    for _ in _:
        pass
    print(' # Finished viewing all data.')


def main():
    parser = argparse.ArgumentParser(description='Convert SUN-RGBD Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default=r'C:\Databases\Kinect',
                        help='Path to the waymo dataset')
    parser.add_argument('--output_path', '-o', default=r'C:\Databases\Kinect_converted',
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
