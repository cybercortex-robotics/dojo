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
Script for converting Mapillary MTSD Dataset to CyberCortex.AI format

Dataset link:
    https://www.mapillary.com/dataset/trafficsign

This script converts train and val dbs.
For test, refer to photos_converter.
"""

import argparse
import os.path
import time
import cv2
import json
import numpy as np
from data.converters.CyC_DatabaseFormat import CyC_DataBase

"""
Dataset Info:

# Tree:
    images
        <>.jpg
    mtsd_v2_fully_annotated
        annotations
            <>.json
        splits
            train.txt
            val.txt
            test.txt
        README.md
        requirements.txt
        visualize_example.py
    mtsd_v2_partially_annotated
        annotations
            <>.json
        splits
            train.txt
    LICENSE.txt

# Anno json format:
{
  "width": 3264,
  "height": 2448,
  "ispano": false,
  "objects": [
    {
      "key": "hxig5ilovzu3c48e4rg5cy",
      "label": "warning--traffic-merges-right--g1",
      "bbox": {
        "xmin": 1529.203125,
        "ymin": 1829.42578125,
        "xmax": 1602.515625,
        "ymax": 1904.1328125
      },
      "properties": {
        "barrier": false,
        "occluded": false,
        "out-of-frame": false,
        "exterior": false,
        "ambiguous": false,
        "included": false,
        "direction-or-information": false,
        "highway": false,
        "dummy": false
      }
    },
    ...
}
"""


def convert_data(args):
    # Count images
    total_images = 0
    try:
        with open('{}/mtsd_v2_{}_annotated/splits/{}.txt'.format(args.path, args.type, args.split)) as text_file:
            for _ in text_file:
                total_images += 1
    except FileNotFoundError:
        print('File {} could not be found.'.format(args.split + '.txt'))

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    # Create empty database
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='image', filter_id=1, filter_name='CameraFront')
    db.add_stream(filter_type='2D_obj_det', filter_id=2, filter_name='Signs2DDet', input_sources=[1])
    # db.show_packing_info()
    db.create_db()

    # Prepare timestamp
    ts_start = int(time.time())
    ts_stop = ts_start
    sampling_time = 5

    # Receive data
    with open('{}/mtsd_v2_{}_annotated/splits/{}.txt'.format(args.path, args.type, args.split)) as text_file:
        for idx, name in enumerate(text_file):
            name = name.strip()

            # Get image
            img_path = '{}/images/{}.jpg'.format(args.path, name)
            if not os.path.isfile(img_path):
                print('File {} not found.'.format(name + '.jpg'))
                continue

            # Get annotation
            try:
                with open('{}/mtsd_v2_{}_annotated/annotations/{}.json'.format(args.path, args.type, name), 'r') as ann:
                    anno = json.load(ann)
            except FileNotFoundError:
                print('File {} not found.'.format(name + '.json'))
                continue

            # Process data
            roi_id, cls, x, y, width, height = [], [], [], [], [], []
            for i, obj in enumerate(anno["objects"]):
                bbox = obj["bbox"]
                roi_id.append(i)
                cls.append(0)  # No class
                x.append(bbox['xmin'])
                y.append(bbox['ymin'])
                width.append(bbox['xmax'] - bbox['xmin'])
                height.append(bbox['ymax'] - bbox['ymin'])

            # Pack data
            data = {
                'datastream_1': {  # image
                    'name': '{}.jpg'.format(ts_stop),
                    'image': img_path
                },
                'datastream_2': {  # 2D_det
                    'frame_id': idx,
                    'roi_id': roi_id,
                    'cls': cls,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                }
            }

            # Add data to database
            db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

            # Advance timestamp
            ts_start = ts_stop
            ts_stop += sampling_time

            # Print progress
            if idx % 25 == 0:
                print('Done {0} out of {1}.'.format(idx, total_images))
    print('Finished process.')


def view_data(args):
    # Count images
    total_images = 0
    try:
        with open('{}/mtsd_v2_{}_annotated/splits/{}.txt'.format(args.path, args.type, args.split)) as text_file:
            for _ in text_file:
                total_images += 1
    except FileNotFoundError:
        print('File {} could not be found.'.format(args.split + '.txt'))

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    # View data
    with open('{}/mtsd_v2_{}_annotated/splits/{}.txt'.format(args.path, args.type, args.split)) as text_file:
        for idx, name in enumerate(text_file):
            name = name.strip()

            # Get image
            img_path = '{}/images/{}.jpg'.format(args.path, name)
            if not os.path.isfile(img_path):
                print('File {} not found.'.format(name + '.jpg'))
                continue
            img = cv2.imread(img_path)

            # Get annotation
            try:
                with open('{}/mtsd_v2_{}_annotated/annotations/{}.json'.format(args.path, args.type, name), 'r') as ann:
                    anno = json.load(ann)
            except FileNotFoundError:
                print('File {} not found.'.format(name + '.json'))
                continue

            # Draw bboxes
            for obj in anno["objects"]:
                bbox = obj["bbox"]
                img = cv2.line(img, (int(bbox['xmin']), int(bbox['ymin'])),
                               (int(bbox['xmax']), int(bbox['ymin'])), (255, 0, 0), 2)
                img = cv2.line(img, (int(bbox['xmax']), int(bbox['ymin'])),
                               (int(bbox['xmax']), int(bbox['ymax'])), (255, 0, 0), 2)
                img = cv2.line(img, (int(bbox['xmax']), int(bbox['ymax'])),
                               (int(bbox['xmin']), int(bbox['ymax'])), (255, 0, 0), 2)
                img = cv2.line(img, (int(bbox['xmin']), int(bbox['ymax'])),
                               (int(bbox['xmin']), int(bbox['ymin'])), (255, 0, 0), 2)

            # Resize image
            img = cv2.resize(img, (1440, 600))

            # Show final image
            cv2.imshow('Images', img)
            cv2.setWindowTitle("Images", '{}/{} : {}'.format(idx+1, total_images, name))
            cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(description='Convert Mapillary MTSD Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default='C:/dev/Databases/mapillary_mtsd',
                        help='Path to the Mapillary MTSD dataset')
    parser.add_argument('--output_path', '-o', default='C:/dev/Databases/mapillary_mtsd/converted_val',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--type', default='fully',
                        help='Annotations type: fully / partially')
    parser.add_argument('--split', default='val',
                        help='train / val')

    parser.add_argument('--mode', '-m', default='convert', help='view / convert')

    args = parser.parse_args()

    # Check arguments
    if args.type not in ['fully', 'partially']:
        raise argparse.ArgumentTypeError(
            "Type is invalid. Try from the following: fully, partially.")
    if args.split not in ['train', 'val']:
        raise argparse.ArgumentTypeError(
            "Split is invalid. Try from the following: train, val.")

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        raise argparse.ArgumentTypeError(
            'Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(args.mode))


if __name__ == '__main__':
    main()
