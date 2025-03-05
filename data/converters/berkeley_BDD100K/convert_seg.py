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
Script for converting Berkeley BDD100K Seg Dataset to CyberCortex.AI format

Dataset Info:
    https://doc.bdd100k.com/download.html#

Dataset link:
    https://bdd-data.berkeley.edu/portal.html#download
"""

import argparse
import time
import cv2
import os
import json
import numpy as np
from data.converters.CyC_DatabaseFormat import *
from data.converters.berkeley_BDD100K.utils import *

"""
Tree:
    bdd100k
        images
            100k
                test / <>.jpg
                train / <>.jpg
                val / <>.jpg
        labels
            sem_seg
                colormaps
                    train / <>.png
                    val / <>.png
                masks
                    train / <>.png
                    val / <>.png
                polygons
                    sem_seg_train.json
                    sem_seg_val.json
                rles
                    sem_seg_train.json
                    sem_seg_val.json
"""


# def test_images():
#     # Check how many labels have correspondent in the rgb images
#     imgs_path = r'C:\dev\Databases\BDD100K\bdd100k\images\100k\train'
#     json_path = r'C:\dev\Databases\BDD100K\bdd100k\labels\pan_seg\polygons\pan_seg_train.json'
#
#     images = os.listdir(imgs_path)
#     with open(json_path, 'r') as json_file:
#         poly = json.load(json_file)
#
#     found = 0
#     total = 0
#     for pol in poly:
#         name = ''.join(pol['name'].split('.')[:-1])
#         total += 1
#         if '{}.jpg'.format(name) in images:
#             found += 1
#     print(total)
#     print(found)
#
#     print('Done')


def convert_data(args):
    # Count data
    images_path = '{}/images/100k/{}'.format(args.path, args.set)
    seg_path = '{}/labels/sem_seg/masks/{}'.format(args.path, args.set)
    total_images = len(os.listdir(seg_path))

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    if args.limit > 0:
        total_images = min(total_images, args.limit)
        print('Converting only {} images.'.format(total_images))

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
    obj_cls = create_obj_cls_file(classes)
    db.add_custom(filter_id=2, data=obj_cls, name='object_classes.conf')

    # Read json
    poly_seg_path = '{}/labels/sem_seg/polygons/sem_seg_{}.json'.format(args.path, args.set)
    with open(poly_seg_path, 'r') as json_file:
        poly_seg = json.load(json_file)

    # Receive data
    images_list = os.listdir(seg_path)
    count_data = 0
    for idx in range(total_images):
        name = ''.join(images_list[idx].split('.')[:-1])

        # Check paths
        if not os.path.isfile('{}/{}.jpg'.format(images_path, name)):
            continue
        if not os.path.isfile('{}/{}.png'.format(seg_path, name)):
            continue

        # Get image size
        seg_img = cv2.imread('{}/{}.png'.format(seg_path, name))[:, :, 0]
        img_size = seg_img.shape

        # Process Polygons
        shape_id, cls_seg, points, pts_list = [], [], [], []
        for p_id, poly in enumerate(poly_seg):
            if name in poly['name']:
                for l_id, label in enumerate(poly['labels']):
                    shape_id.append(l_id)
                    cls_seg.append(name2class(label['category']))
                    pts_list.append([elem for elem in label['poly2d'][0]['vertices']])
                    points.append(pts2str(pts_list[-1]))
                poly_seg.pop(p_id)
                break

        # Generate semseg from polygons
        instance, seg_img, inst_img = generate_semseg(pts_list, cls_seg, img_size)

        # Prepare data
        data = {
            'datastream_1': {  # image
                'name': '{}.jpg'.format(ts_stop),
                'image': '{}/{}.jpg'.format(images_path, name)
            },
            'datastream_2': {  # sem_seg
                'name': '{}.png'.format(ts_stop),
                'semantic': seg_img,
                'instances': inst_img,
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
        count_data += 1
        if count_data % 10 == 0:
            print('Done {0} out of {1}.'.format(count_data, total_images))
    print('\nFinished converting data.\n')


def view_data(args):
    # Count data
    images_path = '{}/images/100k/{}'.format(args.path, args. set)
    seg_path = '{}/labels/sem_seg/colormaps/{}'.format(args.path, args. set)
    inst_path = '{}/labels/ins_seg/colormaps/{}'.format(args.path, args. set)
    total_images = len(os.listdir(seg_path))

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    # View Data
    for idx, seg_name in enumerate(os.listdir(seg_path)):
        name = seg_name.split('.')[0]

        # Check paths
        if not os.path.isfile('{}/{}.jpg'.format(images_path, name)):
            continue
        if not os.path.isfile('{}/{}.png'.format(seg_path, name)):
            continue
        if not os.path.isfile('{}/{}.png'.format(inst_path, name)):
            continue

        # Read images
        rgb_img = cv2.imread('{}/{}.jpg'.format(images_path, name))
        semseg_img = cv2.imread('{}/{}.png'.format(seg_path, name))
        inst_img = cv2.imread('{}/{}.png'.format(inst_path, name))

        final_img = cv2.vconcat([rgb_img, semseg_img, inst_img])
        final_img = cv2.resize(final_img, (1000, 700))

        cv2.imshow('image', final_img)
        cv2.setWindowTitle("image", 'Image {0}/{1} : {2}'.format(idx, total_images, name))
        cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(description='Convert Berkeley BDD100K Seg Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default=r'C:\dev\Databases\BDD100K\bdd100k',
                        help='Path to the original dataset')
    parser.add_argument('--output_path', '-o', default=r'C:\dev\Databases\BDD100K\bdd100k_seg_converted',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--set', default='train',
                        help='train / val')
    parser.add_argument('--limit', default=-1, type=int,
                        help='Number of images to be converted. Set to -1 to convert all db.')
    # parser.add_argument('--width', default=-1, type=int,
    #                     help='Width of images. Set to -1 to not resize.')
    # parser.add_argument('--height', default=-1, type=int,
    #                     help='Height of images. Set to -1 to not resize.')

    parser.add_argument('--mode', '-m', default='view', help='view / convert')

    args = parser.parse_args()

    # Check arguments
    if args.set not in ['train', 'val']:
        raise argparse.ArgumentTypeError(
            "Set {} is invalid. Try train or val.".format(args.set))

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(parser.parse_args().mode))


if __name__ == '__main__':
    main()
