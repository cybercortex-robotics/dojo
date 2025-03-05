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
Script for converting CULane Dataset to CyberCortex.AI format

Dataset link:
    https://xingangpan.github.io/projects/CULane.html
"""

import argparse
import os
import time
import cv2
import glob
import numpy as np
from data.converters.CyC_DatabaseFormat import CyC_DataBase

"""
Dataset info:

Size: ~44 GB

Images + lanes count (only valid, minimum one lane per image):
 - total: 118.110
 - 23_30: 55.437
 - 37_30: 2.835
 - 100_30: 15.842
 - 161_90: 16.210
 - 182_30: 15.710
 - 193_90: 12.076

Splits:
 - train: driver_23_30frame, driver_161_90frame, driver_182_30frame
 - val: driver_23_30frame
 - test: driver_100_30frame, driver_193_90frame, driver_37_30frame
 
Data categorisation:
 - only rgb: video_example.zip (05081544_0305)
 - rgb + lanes: 37_30, 100_30, 193_90
 - rgb + gt lanes: 23_30, 161_90, 182_30
 
Components:
 - driver_<>_<>frame - folders containing rgb + lane information
 - laneseg_label_w16 - segmentation for the lanes for some of the drivers (not used here)
 - list - text files with paths towards the data

Lists:
 - list/train.txt format: - images
    /driver_23_30frame/05151649_0422.MP4/00000.jpg
    /driver_23_30frame/05151649_0422.MP4/00030.jpg
 - list/train_gt.txt format: - image + ground truth + ?
    /driver_23_30frame/05151649_0422.MP4/00000.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00000.png 1 1 1 1
    /driver_23_30frame/05151649_0422.MP4/00030.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00030.png 1 1 1 1
"""


def create_seg(img_size, anno_path, thickness):
    seg_img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    inst_img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

    with open(anno_path, 'r') as f:
        anno_file = f.read().splitlines()

    # If file does not contain even one line, return None
    # if len(anno_file) == 0:
    #     return None, None

    lane_id = 1
    for anno in anno_file:
        pts_list = [anno.split(' ')[i:i + 2] for i in range(0, len(anno.split(' ')), 2)]
        for i in range(1, len(pts_list[:-1])):
            cv2.line(seg_img, (int(float(pts_list[i - 1][0])), int(float(pts_list[i - 1][1]))),
                     (int(float(pts_list[i][0])), int(float(pts_list[i][1]))),
                     (1, 1, 1), thickness=thickness)
            cv2.line(inst_img, (int(float(pts_list[i - 1][0])), int(float(pts_list[i - 1][1]))),
                     (int(float(pts_list[i][0])), int(float(pts_list[i][1]))),
                     (lane_id, lane_id, lane_id), thickness=thickness)
        lane_id += 1

    return seg_img, inst_img


def convert_data(args):
    # Count data
    print('Converting data from the following drivers: {}'.format(' '.join(args.drv_list)))

    # Dataset index
    db_idx = 1

    # Receive data
    for driver in args.drv_list:
        stream_list = os.listdir('{}/{}'.format(args.path, driver))
        print('\n # Starting conversion for {} that contains {} streams.'.format(driver, len(stream_list)))
        for stream in stream_list:
            # Create empty database with the index db_idx for every stream
            db = CyC_DataBase(db_path='{}/DB_{}'.format(args.output_path, db_idx), core_id=1)
            db.add_stream(filter_type='image', filter_id=1, filter_name='CameraFront')
            db.add_stream(filter_type='lane', filter_id=2, filter_name='LanesFront', input_sources=[1])
            # db.show_packing_info()
            db.create_db()

            # Prepare timestamp
            ts_start = int(time.time())
            ts_stop = ts_start
            sampling_time = 5

            images = [os.path.basename(x) for x in glob.glob(f'{args.path}/{driver}/{stream}/*.jpg')]
            img_index = 0
            for image in images:
                img_name = image.split('.')[0]

                # If not lane annotation, skip img
                if not os.path.isfile(f'{args.path}/{driver}/{stream}/{img_name}.lines.txt'):
                    continue

                # Process data
                rgb_img = cv2.imread(f'{args.path}/{driver}/{stream}/{image}')
                seg_img, inst_img = create_seg(rgb_img.shape,
                                               f'{args.path}/{driver}/{stream}/{img_name}.lines.txt',
                                               args.thickness)

                if seg_img is None:
                    continue

                # Pack data
                data = {
                    'datastream_1': {  # image
                        'name': '{}.jpg'.format(ts_stop),
                        'image': cv2.resize(rgb_img, [args.width, args.height])
                    },
                    'datastream_2': {  # lane_seg
                        'name': '{}.png'.format(ts_stop),
                        'semantic': cv2.resize(seg_img, [args.width, args.height]),
                        'instances': cv2.resize(inst_img, [args.width, args.height]),
                        'lane_id': 0,
                        'points': [],  # TODO?
                        'theta_0': '',
                        'theta_1': '',
                        'theta_2': '',
                        'theta_3': ''
                    }
                }

                # Add data to database
                db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

                # Advance timestamp
                ts_start = ts_stop
                ts_stop += sampling_time

                img_index += 1
            db_idx += 1
            print(' - Finished {}/{} with {} images.'.format(driver, stream, img_index))
        print(' - Finished converting {}.'.format(driver))


def view_data(args):
    # Count data
    print('Viewing data from the following drivers: {}'.format(' '.join(args.drv_list)))

    # View data
    for driver in args.drv_list:
        stream_list = os.listdir('{}/{}'.format(args.path, driver))
        print('\n # Starting viewing of {} that contains {} streams.'.format(driver, len(stream_list)))
        for stream in stream_list:
            images = [os.path.basename(x) for x in glob.glob(f'{args.path}/{driver}/{stream}/*.jpg')]
            img_index = 0
            for image in images:
                img_name = image.split('.')[0]

                # If not lane annotation, skip img
                if not os.path.isfile(f'{args.path}/{driver}/{stream}/{img_name}.lines.txt'):
                    continue

                # Process data
                rgb_img = cv2.imread(f'{args.path}/{driver}/{stream}/{image}')
                seg_img, inst_img = create_seg(rgb_img.shape,
                                               f'{args.path}/{driver}/{stream}/{img_name}.lines.txt',
                                               args.thickness)

                if seg_img is None:
                    continue

                final_image = cv2.addWeighted(rgb_img, 0.8, seg_img*255, 0.5, 0.0)

                final_image = cv2.resize(final_image, [args.width, args.height])

                cv2.imshow('image', final_image)
                cv2.setWindowTitle("image", f'{driver}/{stream}/{img_name}')
                cv2.waitKey(2)

                img_index += 1

            print(' - Finished {}/{} with {} images.'.format(driver, stream, img_index))
        print(' - Finished viewing {}.'.format(driver))
    print(' # Finished viewing all data.')


def main():
    parser = argparse.ArgumentParser(description='Convert CULane Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default=r'C:\dev\Databases\CULane',
                        help='Path to the culane dataset')
    parser.add_argument('--output_path', '-o', default=r'C:\dev\Databases\CULane',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--drivers', '-d', default='all',
                        help='Drivers to convert. Example: \'23_30,182_30\' or just \'all\' to convert all.')
    parser.add_argument('--width', default=800, type=int,
                        help='New width of the output image. Set to -1 to keep original size.')
    parser.add_argument('--height', default=600, type=int,
                        help='New height of the output image. Set to -1 to keep original size.')
    parser.add_argument('--thickness', default=5, type=int,
                        help='Thickness for the lane segmentation.')

    parser.add_argument('--mode', '-m', default='view', help='view / convert')

    args = parser.parse_args()

    all_drivers = ['driver_23_30frame', 'driver_37_30frame', 'driver_100_30frame',
                   'driver_161_90frame', 'driver_182_30frame', 'driver_193_90frame']

    # Prepare drivers list
    args.drv_list = []
    if args.drivers == 'all':
        args.drv_list = all_drivers
    else:
        for drv in args.drivers.split(','):
            drv_name = 'driver_{}frame'.format(drv)
            if drv_name in all_drivers:
                if drv_name not in args.drv_list:
                    args.drv_list.append(drv_name)
            else:
                print('{} not found.'.format(drv_name))
    if len(args.drv_list) == 0:
        print('Driver list empty.')
        exit()

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(parser.parse_args().mode))


if __name__ == '__main__':
    main()
