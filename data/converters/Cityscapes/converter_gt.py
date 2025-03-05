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
Script for converting Cityscapes Ground truth Dataset to CyberCortex.AI format

Cityscapes dataset:
    https://www.cityscapes-dataset.com/downloads/

Databases needed:
    gtFine_trainvaltest.zip or gtCoarse.zip
    leftImg8bit_trainvaltest.zip or leftImg8bit_trainextra.zip

Contains:
    gtFine: train, test, val
    gtCoarse: train, train_extra, val

Info:
    Instance image does not identify object from the same class differently!
"""

import argparse
import os
import time
import cv2
import json
import numpy as np
from data.converters.CyC_DatabaseFormat import *
from data.converters.Cityscapes.utils import *

"""
Dataset Info:

Tree:
    gtFine
        test
            <city>
                <city>_000000_000019_gtFine_color.png
                <city>_000000_000019_gtFine_instanceIds.png
                <city>_000000_000019_gtFine_labelIds.png
                <city>_000000_000019_gtFine_polygons.json
                ...
            ...
        train
            <city>
                <city>_000000_000019_gtFine_color.png
                <city>_000000_000019_gtFine_instanceIds.png
                <city>_000000_000019_gtFine_labelIds.png
                <city>_000000_000019_gtFine_polygons.json
                ...
            ...
        val
            <city>
                <city>_000000_000019_gtFine_color.png
                <city>_000000_000019_gtFine_instanceIds.png
                <city>_000000_000019_gtFine_labelIds.png
                <city>_000000_000019_gtFine_polygons.json
                ...
            ...
    leftImg8bit
        test
            <city>
                <city>_000000_000019_leftImg8bit.png
                ...
            ...
        train
            <city>
                <city>_000000_000019_leftImg8bit.png
                ...
            ...
        val
            <city>
                <city>_000000_000019_leftImg8bit.png
                ...
            ...
"""


def create_obj_cls():
    classes_list = []

    for cls in classes:
        classes_list.append({
            'name': cls['name'].replace(' ', '_'),
            'countable': cls['hasInstances']
        })

    return create_obj_cls_file(classes_list)


def pts2str(pts, dim=(-1, -1), im_size=(-1, -1)):
    if dim[0] != -1 and dim[1] != -1:
        rate_x = dim[0]/im_size[0]
        rate_y = dim[1]/im_size[1]
        str_poly = "[{}]".format(
            "".join(["[{} {}]".format(x1 * rate_x, x2 * rate_y) for x1, x2 in pts])
        )
    else:
        str_poly = "[{}]".format(
            "".join(["[{} {}]".format(x1, x2) for x1, x2 in pts])
        )
    return str_poly


def convert_data(args):
    # Prepare paths
    anno_path = '{}/{}/{}'.format(args.path, args.type, args.set)
    imgs_path = '{}/leftImg8bit/{}'.format(args.path, args.set)

    # Check paths
    if not os.path.isdir(anno_path):
        print('Gt path for {} with {} is not valid'.format(args.type, args.set))
    if not os.path.isdir(imgs_path):
        print('Images path for {} with {} is not valid'.format(args.type, args.set))

    # Read cities
    cities = os.listdir(anno_path)
    if args.cities != 'all':
        cities_aux = []
        for city in args.cities.split(','):
            if city not in cities:
                print('City {} not available.'.format(city))
            else:
                cities_aux.append(city)
        cities = cities_aux

    out_dim = (args.width, args.height)  # Output images size

    # Count images
    total_images = 0
    for city in cities:
        total_images += len(os.listdir('{}/{}'.format(imgs_path, city)))

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    # Create empty database
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='image', filter_id=1, filter_name='CameraFront')
    db.add_stream(filter_type='semseg', filter_id=2, filter_name='SemSegFront', input_sources=[1])
    # db.show_packing_info()
    db.create_db()

    # Prepare timestamp
    ts_start = int(time.time())
    ts_stop = ts_start
    sampling_time = 5

    # Prepare and save object classes file
    obj_cls = create_obj_cls()
    db.add_custom(filter_id=2, data=obj_cls, name='object_classes.conf')

    # Receive data
    current_idx = 0
    for city_idx, city in enumerate(cities):
        for im_idx, image in enumerate(os.listdir('{}/{}'.format(imgs_path, city))):
            # Get name
            name = '_'.join(image.split('.')[0].split('_')[:-1])

            # Prepare paths
            rgb_path = '{}/{}/{}_leftImg8bit.png'.format(imgs_path, city, name)
            seg_path = '{}/{}/{}_gtFine_labelIds.png'.format(anno_path, city, name)
            inst_path = '{}/{}/{}_gtFine_instanceIds.png'.format(anno_path, city, name)

            # Process images
            im_size = (-1, -1)
            if out_dim[0] > 0 and out_dim[1] > 0:
                rgb_raw = cv2.imread(rgb_path)
                im_size = (rgb_raw.shape[1], rgb_raw.shape[0])
                rgb_final = cv2.resize(rgb_raw, out_dim)
                seg_final = cv2.resize(cv2.imread(seg_path), out_dim)
                inst_final = cv2.resize(cv2.imread(inst_path), out_dim)
            else:
                rgb_final = rgb_path
                seg_final = seg_path
                inst_final = inst_path

            # Read process json => polygons
            with open('{}/{}/{}_gtFine_polygons.json'.format(anno_path, city, name), 'r') as file:
                poly_json = json.load(file)

            shape_id, cls_seg, instance, points = [], [], [], []
            for obj_idx, obj in enumerate(poly_json["objects"]):
                shape_id.append(obj_idx)
                cls_idx = name2cls(obj['label'])
                cls_seg.append(cls_idx if cls_idx != -1 else 0)
                instance.append(0)  # No instance id
                points.append(pts2str(obj['polygon'], out_dim, im_size))

            # Pack data
            data = {
                'datastream_1': {  # image
                    'name': '{}.jpg'.format(ts_stop),
                    'image': rgb_final
                },
                'datastream_2': {  # sem_seg
                    'name': '{}.png'.format(ts_stop),
                    'semantic': seg_final,
                    'instances': inst_final,
                    'shape_id': shape_id,
                    'cls': cls_seg,
                    'instance': instance,
                    'points': points
                }
            }

            # Add data to database
            db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

            # Advance timestamp
            ts_start = ts_stop
            ts_stop += sampling_time

            # Print progress
            current_idx += 1
            if current_idx % 25 == 0:
                print('Done {0} out of {1}.'.format(current_idx, total_images))
    print('Finished process.')


def view_data(args):
    # Prepare paths
    anno_path = '{}/{}/{}'.format(args.path, args.type, args.set)
    imgs_path = '{}/leftImg8bit/{}'.format(args.path, args.set)

    # Check paths
    if not os.path.isdir(anno_path):
        print('Gt path for {} with {} is not valid'.format(args.type, args.set))
    if not os.path.isdir(imgs_path):
        print('Images path for {} with {} is not valid'.format(args.type, args.set))

    # Read cities
    cities = os.listdir(anno_path)
    if args.cities != 'all':
        cities_aux = []
        for city in args.cities.split(','):
            if city not in cities:
                print('City {} not available.'.format(city))
            else:
                cities_aux.append(city)
        cities = cities_aux

    # Count images
    total_images = 0
    for city in cities:
        total_images += len(os.listdir('{}/{}'.format(imgs_path, city)))

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    # View data
    current_idx = 0
    for city_idx, city in enumerate(cities):
        for im_idx, image in enumerate(os.listdir('{}/{}'.format(imgs_path, city))):
            # Get name
            name = '_'.join(image.split('.')[0].split('_')[:-1])

            # Read images
            rgb_img = cv2.imread('{}/{}/{}_leftImg8bit.png'.format(imgs_path, city, name))
            seg_img = cv2.imread('{}/{}/{}_gtFine_color.png'.format(anno_path, city, name))

            final_img = cv2.vconcat([rgb_img, seg_img])
            final_img = cv2.resize(final_img, (1440, 600))

            # Show final image
            cv2.imshow('Images', final_img)
            current_idx += 1
            cv2.setWindowTitle("Images", '{}/{} : {} : {}'.
                               format(current_idx, total_images, city, image))
            cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(description='Convert Cityscapes Ground truth Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default='C:/dev/Databases/Cityscapes',
                        help='Path to the original dataset')
    parser.add_argument('--output_path', '-o', default='C:/dev/Databases/Cityscapes/gt_converted',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--type', default='gtFine',
                        help='gtFine / gtCoarse')
    parser.add_argument('--cities', default='bremen,hamburg,ulm',
                        help='Cities to convert. Set to \'all\' to convert all.')
    parser.add_argument('--set', default='train',
                        help='train / train_extra / test / val')
    parser.add_argument('--width', default=-1, type=int,
                        help='Width of images. Set to -1 to not resize.')
    parser.add_argument('--height', default=-1, type=int,
                        help='Height of images. Set to -1 to not resize.')

    parser.add_argument('--mode', '-m', default='convert', help='view / convert')

    args = parser.parse_args()

    # Check arguments
    if args.type not in ['gtFine', 'gtCoarse']:
        raise argparse.ArgumentTypeError(
            "Type {} is invalid. Try gtFine or gtCoarse.".format(args.type))
    if args.set not in ['train', 'train_extra', 'test', 'val']:
        raise argparse.ArgumentTypeError(
            "Set {} is invalid. Try train, val or test.".format(args.set))

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(args.mode))


if __name__ == '__main__':
    main()
