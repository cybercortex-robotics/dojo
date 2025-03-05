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
Script for converting EuroCity Persons dataset to CyberCortex.AI format

Dataset link:
    https://eurocity-dataset.tudelft.nl/
    
To save only selected classes, edit in utils in method "check_data" with classes from above.
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
Dataset examples:
    https://eurocity-dataset.tudelft.nl/eval/overview/examples

Tree:
    ECP:
        day:
            img:
                train:
                    <city_name>:
                        <city_name>_00000.png
                        <city_name>_00001.png
                        ...
                    <another_city_name>:
                        ...
                    ...
                val:
                    ...
                test:
                    ...
            labels:
                train:
                    <city_name>:
                        <city_name>_00000.json
                        <city_name>_00001.json
                        ...
                    ...
                ...
        night:
            ...
        
Json format:
{
    "tags": [], 
    "imageheight": 1024, 
    "imagewidth": 1920, 
    "children": [
        {
            "tags": [], 
            "x0": 951, 
            "y1": 576, 
            "y0": 547, 
            "x1": 960, 
            "children": [], 
            "identity": "person-group-far-away"
        }, 
        {
            ...
        }
    ...
    ],
    "identity": "frame"
}
"""


def convert_data(args):
    img_global_folder = args.path + r'\{0}\img\{1}'
    labels_global_folder = args.path + r'\{0}\labels\{1}'

    # Count data
    total_images = 0
    for ds_time, ds_type in [elem.split('-') for elem in args.datasets.split(',')]:
        img_folder = img_global_folder.format(ds_time, ds_type)

        if not os.path.exists(img_folder):
            continue

        locations = os.listdir(img_folder)

        for loc in locations:
            total_images += len(os.listdir(os.path.join(img_folder, loc)))
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
    obj_cls = create_obj_cls()
    db.add_custom(filter_id=2, data=obj_cls, name='object_classes.conf')

    # Receive data
    image_count = 0
    frame_id = 0
    for ds_time, ds_type in [elem.split('-') for elem in args.datasets.split(',')]:
        img_folder = img_global_folder.format(ds_time, ds_type)
        labels_folder = labels_global_folder.format(ds_time, ds_type)

        if not os.path.exists(img_folder):
            continue
        if ds_type != 'test' and not os.path.exists(labels_folder):
            continue

        locations = os.listdir(img_folder)

        new_size = ()
        if args.resize:
            new_size = (int(args.width), int(args.height))

        for loc in locations:
            imgs_name = os.listdir(os.path.join(img_folder, loc))

            for img_name in imgs_name:
                img_path = os.path.join(img_folder, loc, img_name)
                json_path = os.path.join(labels_folder, loc, img_name.split('.')[0] + '.json')

                with open(json_path) as json_file:
                    json_data = json.load(json_file)

                    bboxes = []
                    for item in json_data['children']:
                        bboxes += read_bboxes(item)

                # Take every bounding box and add it all on the same frame / timestamp
                roi_id, cls, x, y, w, h = [], [], [], [], [], []
                for bbox_id, bbox in enumerate(bboxes):

                    # Filter
                    if not check_data(bbox):
                        continue

                    roi_id.append(bbox_id)
                    cls.append(bbox['cls'])
                    x.append(bbox['x'])
                    y.append(bbox['y'])
                    w.append(bbox['w'])
                    h.append(bbox['h'])

                if len(roi_id) != 0:
                    if args.resize:
                        img = cv2.imread(img_path)

                        # Resize img and detections
                        old_size = (json_data['imagewidth'], json_data['imageheight'])
                        ratio_x = new_size[0] / old_size[0]
                        ratio_y = new_size[1] / old_size[1]

                        resized_img = cv2.resize(img, new_size)
                        x = [int(el * ratio_x) for el in x]
                        y = [int(el * ratio_y) for el in y]
                        w = [int(el * ratio_x) for el in w]
                        h = [int(el * ratio_y) for el in h]

                        # Prepare data
                        data = {
                            'datastream_1': {  # image
                                'name': '{}.png'.format(ts_stop),
                                'image': resized_img
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
                    else:
                        # Prepare data
                        data = {
                            'datastream_1': {  # image
                                'name': '{}.png'.format(ts_stop),
                                'image': img_path
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
    img_global_folder = args.path + r'\{0}\img\{1}'
    labels_global_folder = args.path + r'\{0}\labels\{1}'

    # Count data
    total_images = 0
    for ds_time, ds_type in [elem.split('-') for elem in args.datasets.split(',')]:
        img_folder = img_global_folder.format(ds_time, ds_type)
        # labels_folder = labels_global_folder.format(ds_time, ds_type)

        if not os.path.exists(img_folder):
            continue

        locations = os.listdir(img_folder)

        for loc in locations:
            total_images += len(os.listdir(os.path.join(img_folder, loc)))
    print('Total images found: {}'.format(total_images))

    # View data
    image_count = 0
    for ds_time, ds_type in [elem.split('-') for elem in args.datasets.split(',')]:
        img_folder = img_global_folder.format(ds_time, ds_type)
        labels_folder = labels_global_folder.format(ds_time, ds_type)

        if not os.path.exists(img_folder):
            continue
        if ds_type != 'test' and not os.path.exists(labels_folder):
            continue

        locations = os.listdir(img_folder)

        new_size = ()
        if args.resize:
            new_size = (int(args.width), int(args.height))

        for loc in locations:
            imgs_name = os.listdir(os.path.join(img_folder, loc))

            for img_name in imgs_name:
                img = cv2.imread(os.path.join(img_folder, loc, img_name))
                json_path = os.path.join(labels_folder, loc, img_name.split('.')[0] + '.json')

                if args.resize:
                    img = cv2.resize(img, new_size)

                with open(json_path) as json_file:
                    json_data = json.load(json_file)

                    bboxes = []
                    for item in json_data['children']:
                        bboxes += read_bboxes(item)

                    old_size = (json_data['imagewidth'], json_data['imageheight'])

                bbox_on_image = False
                for bbox in bboxes:

                    # Filter
                    if not check_data(bbox):
                        continue

                    color = colors[bbox['cls']]

                    if args.resize:
                        ratio_x = new_size[0] / old_size[0]
                        ratio_y = new_size[1] / old_size[1]
                        bbox['x'] = int(bbox['x'] * ratio_x)
                        bbox['y'] = int(bbox['y'] * ratio_y)
                        bbox['w'] = int(bbox['w'] * ratio_x)
                        bbox['h'] = int(bbox['h'] * ratio_y)

                    img = cv2.line(img, (bbox['x'], bbox['y']), (bbox['x'] + bbox['w'], bbox['y']),
                                   color, thickness=2)
                    img = cv2.line(img, (bbox['x'] + bbox['w'], bbox['y']),
                                   (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']),
                                   color, thickness=2)
                    img = cv2.line(img, (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']),
                                   (bbox['x'], bbox['y'] + bbox['h']),
                                   color, thickness=2)
                    img = cv2.line(img, (bbox['x'], bbox['y'] + bbox['h']), (bbox['x'], bbox['y']),
                                   color, thickness=2)
                    bbox_on_image = True

                if bbox_on_image:
                    cv2.imshow('image', img)
                    cv2.setWindowTitle("image", 'Image-{0}/{1}-{2}'.format(image_count, total_images, img_name))
                    cv2.waitKey(0)

                # Count images
                image_count += 1
                if image_count % 50 == 0:
                    print('Done {0} out of {1}.'.format(image_count, total_images))


def data_statistics(args):
    img_global_folder = args.path + r'\{0}\img\{1}'
    labels_global_folder = args.path + r'\{0}\labels\{1}'

    # Statistic vars
    num_of_bboxes = 0
    num_of_images_with_ped = 0
    bbox_areas = []
    bbox_areas_dict = {}

    # Count data
    total_images = 0
    for ds_time, ds_type in [elem.split('-') for elem in args.datasets.split(',')]:
        img_folder = img_global_folder.format(ds_time, ds_type)
        # labels_folder = labels_global_folder.format(ds_time, ds_type)

        if not os.path.exists(img_folder):
            continue

        locations = os.listdir(img_folder)

        for loc in locations:
            total_images += len(os.listdir(os.path.join(img_folder, loc)))
    print('Total images found: {}'.format(total_images))

    # Analysis
    image_count = 0
    for ds_time, ds_type in [elem.split('-') for elem in args.datasets.split(',')]:
        img_folder = img_global_folder.format(ds_time, ds_type)
        labels_folder = labels_global_folder.format(ds_time, ds_type)

        if not os.path.exists(img_folder):
            continue
        if ds_type != 'test' and not os.path.exists(labels_folder):
            continue

        locations = os.listdir(img_folder)

        for loc in locations:
            imgs_name = os.listdir(os.path.join(img_folder, loc))

            for img_name in imgs_name:
                json_path = os.path.join(labels_folder, loc, img_name.split('.')[0] + '.json')

                with open(json_path) as json_file:
                    json_data = json.load(json_file)

                    bboxes = []
                    for item in json_data['children']:
                        bboxes += read_bboxes(item)

                bbox_exists = False
                for bbox in bboxes:

                    # Filter
                    if not check_data(bbox):
                        continue

                    bbox_exists = True
                    num_of_bboxes += 1

                    area = bbox['w'] * bbox['h']
                    bbox_areas.append(area)

                if bbox_exists:
                    num_of_images_with_ped += 1

                # Count data
                image_count += 1
                if image_count % 50 == 0:
                    print('Done {0} from {1}.'.format(image_count, total_images))

    sub_areas_size = 100
    for i in range(int(min(bbox_areas)/sub_areas_size), int(max(bbox_areas)/sub_areas_size) + 1):
        val_range = '{}-{}'.format(i*sub_areas_size, (i+1)*sub_areas_size)
        bbox_areas_dict[val_range] = 0

    for area in bbox_areas:
        val = int(area/sub_areas_size)
        val_range = '{}-{}'.format(val*sub_areas_size, (val+1)*sub_areas_size)
        bbox_areas_dict[val_range] += 1

    # Print statistics
    print('\nFrom a total of {0}, found:'.format(total_images))
    print(' - {0} images containing bboxes for a total of {1} bboxes.'.
          format(num_of_images_with_ped, num_of_bboxes))
    print(' - As for the areas of each bbox, statistics:')
    print('   min: {}, max: {}, mean: {:.5f}'.
          format(min(bbox_areas), max(bbox_areas), np.mean(bbox_areas)))

    # Take only some items
    bbox_areas_dict = {k: bbox_areas_dict[k] for k in list(bbox_areas_dict.keys())[:50]}

    plt.barh(list(bbox_areas_dict.keys()), list(bbox_areas_dict.values()))
    plt.title('Statistics of ECP bbox areas.')
    plt.ylabel('Range')
    plt.xlabel('Number of areas')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Convert EuroCity Persons dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default=r'C:\data\ECP',
                        help='Path to the original dataset')
    parser.add_argument('--output_path', '-o', default=r'C:\data\ECP_converted',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--datasets', '-d', default=r'day-train',
                        help='day-train,day-val,night-train...')
    parser.add_argument('--resize', '-r', default=False,
                        help='Resize to a scale image and detections.')
    parser.add_argument('--width', default=640, type=int,
                        help='New width of the output image.')
    parser.add_argument('--height', default=640, type=int,
                        help='New height of the output image.')

    parser.add_argument('--mode', '-m', default='convert', help='view / convert / stats')

    args = parser.parse_args()

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    elif args.mode == 'stats':
        data_statistics(args=args)
    else:
        print('Mode \'{}\' does not exist. Try \'view\' or \'convert\' or \'stats\'.'.format(parser.parse_args().mode))


if __name__ == '__main__':
    main()
