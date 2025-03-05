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
Script for converting TuSimple LanesDetection Train Dataset to CyberCortex.AI format

Dataset link:
    https://github.com/TuSimple/tusimple-benchmark/issues/3
"""

import argparse
import time
import json
import cv2
import numpy as np
from data.converters.CyC_DatabaseFormat import CyC_DataBase

"""
Dataset Info:

# Label data json format:
{
  "lanes": [
    [-2, -2, -2, -2, -2, -2, -2, -2, 499, 484, 470, 453, 435, 418, 400, 374, 346, 318, 290, 262, 235, ...],
    [-2, -2, -2, -2, -2, -2, -2, -2, -2, 529, 531, 533, 536, 538, 540, 540, 538, 536, 531, 525, 519, 513, ...],
    [-2, -2, -2, -2, -2, -2, -2, -2, 553, 568, 583, 598, 613, 640, 667, 693, 719, 740, 761, 783, 804, 825, ...],
    [-2, -2, -2, -2, -2, -2, -2, -2, 558, 585, 613, 646, 679, 714, 770, 817, 865, 912, 954, 994, 1033, ...],
  ],
  "h_samples": [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, ...],
  "raw_file": "clips/0531/1492626287507231547/20.jpg"
}
"""


def convert_data(args):
    # Count data
    images = {}
    for json_file in args.json_files.split(','):
        try:
            with open(args.input_path + '/' + json_file) as f:
                images[json_file] = 0
                for _ in f:
                    images[json_file] += 1
        except FileNotFoundError:
            print('File {} could not be found.'.format(json_file))

    total_images = sum(list(images.values()))
    if len(images.keys()) == 1:
        print('Found one json file containing {} images.\n'.format(total_images))
    elif len(images.keys()) > 1:
        print('Found {0} json files containing {1} images each.'.format(len(images.keys()), list(images.values())))
        print('Total images found: {}\n'.format(total_images))
    else:
        print('No available json files found. Exiting..')
        return

    # Create empty database
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='image', filter_id=1, filter_name='CameraFront')
    db.add_stream(filter_type='sem_seg', filter_id=2, filter_name='LaneFrontSeg', input_sources=[1])
    # db.show_packing_info()
    db.create_db()

    # Prepare timestamp
    ts_start = int(time.time())
    ts_stop = ts_start
    sampling_time = 5

    # Receive data
    image_count = 0
    for json_file in args.json_files.split(','):
        try:
            with open(args.input_path + '/' + json_file) as f:
                for line in f:
                    data = json.loads(line)

                    # Data to save
                    rgb_img = np.array(cv2.imread(args.input_path + '/' + data['raw_file']))
                    seg_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1]))
                    inst_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1]))

                    for lane_id, lane in enumerate(data['lanes']):  # for each lane
                        x = []
                        y = []
                        for i in range(len(lane)):  # for each point of the lane
                            if lane[i] == -2:
                                continue
                            x.append(lane[i])
                            y.append(data['h_samples'][i])

                        for i in range(1, len(x)):
                            cv2.line(seg_img, (x[i-1], y[i-1]), (x[i], y[i]), 1, thickness=10)
                            cv2.line(inst_img, (x[i-1], y[i-1]), (x[i], y[i]), lane_id + 1, thickness=10)

                    # Pack data
                    data = {
                        'datastream_1': {  # image
                            'name': '{}.jpg'.format(ts_stop),
                            'image': rgb_img
                        },
                        'datastream_2': {  # sem_seg
                            'name': '{}.png'.format(ts_stop),
                            'semantic': seg_img,
                            'instances': inst_img,
                            'shape_id': -1,
                            'cls': -1,
                            'instance': -1,
                            'points': []
                        }
                    }

                    # Add data to database
                    db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

                    # Advance timestamp
                    ts_start = ts_stop
                    ts_stop += sampling_time

                    # Print progress
                    image_count += 1
                    if image_count % 25 == 0:
                        print('Done {0} out of {1}.'.format(image_count, total_images))
            print('\nFinished extracting data from json: {}\n'.format(json_file))
        except FileNotFoundError:
            pass

    print('Finished process.')


def view_data(args):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # Count data
    images = {}
    for json_file in args.json_files.split(','):
        try:
            with open(args.input_path + '/' + json_file) as f:
                images[json_file] = 0
                for _ in f:
                    images[json_file] += 1
        except FileNotFoundError:
            print('File {} could not be found.'.format(json_file))

    total_images = sum(list(images.values()))
    print('Total images found: {}'.format(total_images))

    # View Data
    image_count = 0
    for json_file in args.json_files.split(','):
        try:
            with open(args.input_path + '/' + json_file) as f:
                for line in f:
                    image_count += 1
                    data = json.loads(line)

                    image = cv2.imread(args.input_path + '/' + data['raw_file'])
                    for lane_num, lane in enumerate(data['lanes']):
                        for i in range(len(lane)):
                            if lane[i] == -2:
                                continue
                            image = cv2.circle(image, (lane[i], data['h_samples'][i]), radius=7,
                                               color=colors[lane_num], thickness=-1)

                    cv2.imshow('image', image)
                    cv2.setWindowTitle("image", 'Image-{0}/{1}-{2}'.format(image_count, total_images, data['raw_file']))
                    cv2.waitKey(0)
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description='Convert TuSimple Lane Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--input_path', '-p', default='D:/dev/DataBases/TuSimple',
                        help='Path to the tusimple dataset')
    parser.add_argument('--output_path', '-o', default='D:/dev/DataBases/TuSimple/too_simple',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--json_files', default='label_data_0531.json',
                        help='Path to the config json file')

    parser.add_argument('--mode', '-m', default='convert', help='view / convert')

    args = parser.parse_args()

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode {} does not exist. Try \'view\' or \'convert\'.'.format(parser.parse_args().mode))


if __name__ == '__main__':
    main()
