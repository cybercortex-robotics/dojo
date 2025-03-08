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
Script for converting MOT17Det dataset to CyberCortex.AI format

Dataset link:
    https://motchallenge.net/data/MOT17Det/
    
The script will convert only the train folder.
For test, refer to converters/general_purpose/photos2CyC.py
"""

import argparse
import time
import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from data.converters.CyC_DatabaseFormat import CyC_DataBase, create_obj_cls_file
from data.converters.ECP_EuroCity_Persons.utils import *

"""
Tree:
    MOT17:
        test
            MOT17-01
                img1
                    000001.jpg
                    000002.jpg
                    ...
                seqinfo.ini
            MOT17-03
            MOT17-06
            ...
        train
            MOT17-02
                gt
                    gt.txt
                img1
                    000001.jpg
                    000002.jpg
                    ...
                seqinfo.ini
            MOT17-04
            MOT17-05
            ...
            
GroundTruth (gt.txt) format:
<img_id> <det_id> <x> <y> <w> <h> <?> <cls> <conf>
"""


class Idx:
    img_id, det_id = 0, 1
    x, y = 2, 3
    w, h = 4, 5
    cls = 7
    conf = 8


# Object classes
""" 
1 - pedestrian
2 - 
3 - car
4 - bike
5 - 
6 - 
7 - pedestrian sitting ?
8 - statue ?
9 - 
10 - pole
11 - sign/flower/light
"""
classes = [
    {
        'name': 'pedestrian',
        'countable': True
    },
    # {
    #     'name': 'car',
    #     'countable': True
    # },
    # {
    #     'name': 'bike',
    #     'countable': True
    # }
]


def mot2cyc(mot_cls):
    if mot_cls == 1 or mot_cls == 7:
        return 0  # pedestrian
    # if mot_cls == 3:
    #     return 1  # car
    # if mot_cls == 4:
    #     return 2  # bike
    return -1


def convert_data(args):
    imgs_path = args.path + r'\train\{}\img1'
    gt_path = args.path + r'\train\{}\gt\gt.txt'

    # Count data
    total_images = 0
    for video in args.videos:
        img_folder = imgs_path.format(video)

        if not os.path.exists(img_folder):
            continue

        total_images += len(os.listdir(img_folder))
    print('Total images found: {}'.format(total_images))

    # Create empty database
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='image', filter_id=1, filter_name='CameraFront')
    db.add_stream(filter_type='2D_obj_det', filter_id=2, filter_name='ObjectDet', input_sources=[1])
    # db.show_packing_info()
    db.create_db()

    # Prepare timestamp
    ts_start = int(time.time())
    ts_stop = ts_start
    sampling_time = 5

    # Add object classes file
    obj_cls = create_obj_cls_file(classes)
    db.add_custom(filter_id=2, data=obj_cls, name='object_classes.conf')

    # Receive data
    image_count = 0
    frame_id = 0
    new_size = (int(args.width), int(args.height))
    for video in args.videos:
        img_folder = imgs_path.format(video)
        gt_file = gt_path.format(video)

        if not os.path.exists(img_folder) or not os.path.exists(gt_file):
            print("For video {}, img or gt is missing.".format(video))
            continue

        # Read groundtruth
        with open(gt_file, 'r') as f:
            gt_lines = f.readlines()

        for img_name in os.listdir(img_folder):
            img_id = int(img_name.split('.')[0])
            bbox_on_image = False

            img = None
            ratio_x = 1
            ratio_y = 1
            if args.resize:
                img = cv2.imread(os.path.join(img_folder, img_name))
                old_size = img.shape
                img = cv2.resize(img, new_size)

                # Calculate new ratio
                ratio_x = new_size[0] / old_size[1]
                ratio_y = new_size[1] / old_size[0]

            # Take every entity of the current image
            roi_id, cls, x, y, w, h = [], [], [], [], [], []

            for line in gt_lines:
                line_split = line.strip().split(',')
                if int(line_split[Idx.img_id]) == img_id:
                    if float(line_split[Idx.conf]) < args.confidence:
                        continue

                    cyc_cls = mot2cyc(int(line_split[Idx.cls]))
                    if cyc_cls == -1:
                        continue

                    bbox_on_image = True

                    # Add data
                    roi_id.append(int(line_split[Idx.det_id]))
                    cls.append(cyc_cls)
                    x.append(int(int(line_split[Idx.x]) * ratio_x))
                    y.append(int(int(line_split[Idx.y]) * ratio_y))
                    w.append(int(int(line_split[Idx.w]) * ratio_x))
                    h.append(int(int(line_split[Idx.h]) * ratio_y))

            # If image empty, ignore it
            if not bbox_on_image:
                image_count += 1
                continue

            # Prepare data
            data = {
                'datastream_1': {  # image
                    'name': '{}.png'.format(ts_stop),
                    'image': img if img is not None else os.path.join(img_folder, img_name)
                },
                'datastream_2': {  # 2D_det
                    'frame_id': frame_id,
                    'roi_id': roi_id,
                    'cls': cls,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                }
            }

            # Add data to database
            db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)
            frame_id += 1

            # Advance timestamp
            ts_start = ts_stop
            ts_stop += sampling_time

            # Print progress
            image_count += 1
            if image_count % 25 == 0:
                print('Done {0} out of {1}.'.format(image_count, total_images))
    print('\nFinished converting the data.\n')


def view_data(args):
    colors = [(165, 42, 42), (0, 192, 0), (31, 170, 250), (250, 170, 32), (196, 196, 196),
              (20, 153, 53), (250, 0, 0), (10, 120, 250), (250, 170, 33), (250, 170, 34),
              (128, 128, 128), (250, 170, 35), (102, 102, 156), (128, 64, 255), (140, 140, 200),
              (170, 170, 170), (250, 170, 36), (250, 170, 160), (55, 250, 37), (96, 96, 96)]
    imgs_path = args.path + r'\train\{}\img1'
    gt_path = args.path + r'\train\{}\gt\gt.txt'

    # Count data
    total_images = 0
    for video in args.videos:
        img_folder = imgs_path.format(video)

        if not os.path.exists(img_folder):
            continue

        total_images += len(os.listdir(img_folder))
    print('Total images found: {}'.format(total_images))

    # View data
    image_count = 0
    new_size = (int(args.width), int(args.height))
    for video in args.videos:
        img_folder = imgs_path.format(video)
        gt_file = gt_path.format(video)

        if not os.path.exists(img_folder) or not os.path.exists(gt_file):
            print("For video {}, img or gt is missing.".format(video))
            continue

        # Read groundtruth
        with open(gt_file, 'r') as f:
            gt_lines = f.readlines()

        for img_name in os.listdir(img_folder):
            img_id = int(img_name.split('.')[0])
            img = cv2.imread(os.path.join(img_folder, img_name))
            old_size = img.shape

            # Calculate new ratio
            ratio_x = new_size[0] / old_size[1]
            ratio_y = new_size[1] / old_size[0]

            if args.resize:
                img = cv2.resize(img, new_size)

            for line in gt_lines:
                line_split = line.strip().split(',')
                if int(line_split[Idx.img_id]) == img_id:
                    if float(line_split[Idx.conf]) < args.confidence:
                        continue

                    cyc_cls = mot2cyc(int(line_split[Idx.cls]))
                    if cyc_cls == -1:
                        continue

                    if not args.resize:
                        x = int(line_split[Idx.x])
                        y = int(line_split[Idx.y])
                        w = int(line_split[Idx.w])
                        h = int(line_split[Idx.h])
                    else:
                        x = int(int(line_split[Idx.x]) * ratio_x)
                        y = int(int(line_split[Idx.y]) * ratio_y)
                        w = int(int(line_split[Idx.w]) * ratio_x)
                        h = int(int(line_split[Idx.h]) * ratio_y)
                    color = colors[int(line_split[Idx.det_id]) % len(colors)]

                    # Draw bbox
                    img = cv2.line(img, (x, y), (x + w, y), color, thickness=2)
                    img = cv2.line(img, (x + w, y), (x + w, y + h), color, thickness=2)
                    img = cv2.line(img, (x + w, y + h), (x, y + h), color, thickness=2)
                    img = cv2.line(img, (x, y + h), (x, y), color, thickness=2)
                    # img = cv2.putText(img, f'{line_split[6]},{line_split[7]}', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    #                   1, (0, 0, 255), 2, cv2.LINE_AA)

            # Show image
            cv2.imshow('image', img)
            cv2.setWindowTitle("image", '{0}/{1}'.format(video, img_name))
            cv2.waitKey(0)

            # Print progress
            image_count += 1
            if image_count % 25 == 0:
                print('Done {0} out of {1}.'.format(image_count, total_images))
    print('Done.')


def main():
    parser = argparse.ArgumentParser(description='Convert MOT17Det dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default=r'C:\dev\Databases\MOT17',
                        help='Path to the original dataset')
    parser.add_argument('--output_path', '-o', default=r'C:\dev\Databases\MOT17_converted',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--videos', default=r'*',
                        help='Combinations of 02,04,05,09,10,11,13 or just all: *')
    parser.add_argument('--resize', '-r', default=True,
                        help='Resize to a scale image and detections.')
    parser.add_argument('--width', default=640, type=int,
                        help='New width of the output image.')
    parser.add_argument('--height', default=480, type=int,
                        help='New height of the output image.')
    parser.add_argument('--confidence', '-c', default=0.2, type=float,
                        help='The confidence threshold.')

    parser.add_argument('--mode', '-m', default='view', help='view / convert')

    args = parser.parse_args()

    # Train videos only
    all_videos = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']

    # Check videos argument
    videos = []
    if args.videos == '*':
        videos = all_videos
    else:
        for video_id in args.videos.split(','):
            video_name = 'MOT17-' + video_id
            if video_name in all_videos:
                videos.append(video_name)

    if not len(videos):
        print("No videos selected. Aborting..")
        exit()
    args.videos = videos

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(parser.parse_args().mode))


if __name__ == '__main__':
    main()
