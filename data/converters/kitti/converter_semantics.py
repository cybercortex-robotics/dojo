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
Script for converting Kitti Semantics Dataset to CyberCortex.AI format

Kitti dataset:
    https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

Classes:
    Same as Cityscapes
"""

import argparse
import os
import time
import cv2
import numpy as np
from PIL import Image
from data.converters.CyC_DatabaseFormat import CyC_DataBase

"""
Dataset Info:

Tree:
    testing
        image_2
            000000_10.png
            000001_10.png
            ...
    training
        image_2
            000000_10.png
            000001_10.png
            ...
        instance
            000000_10.png
            000001_10.png
            ...
        semantic
            000000_10.png
            000001_10.png
            ...
        semantic_rgb
            000000_10.png
            000001_10.png
            ...
"""


def create_obj_cls():
    from data.converters.Cityscapes.utils import classes
    classes_list = []

    for cls in classes:
        classes_list.append({
            'name': cls['name'].replace(' ', '_'),
            'countable': cls['hasInstances']
        })

    return create_obj_cls_file(classes_list)


def convert_data(args):
    # Count images
    images_names = os.listdir('{}/training/image_2'.format(args.path))
    total_images = len(images_names)

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    if args.limit > 0:
        total_images = min(total_images, args.limit)
        print('Saving {} images.'.format(total_images))

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
    for idx in range(total_images):
        img_name = images_names[idx]

        # Prepare paths
        rgb_path = '{}/training/image_2/{}'.format(args.path, img_name)
        seg_inst_path = '{}/training/instance/{}'.format(args.path, img_name)

        # Process Instance image
        seg_inst_raw = Image.open(seg_inst_path)  # contains instance and segmentation
        seg_inst_array = np.array(seg_inst_raw, dtype=np.uint16)

        semseg_final = np.array(seg_inst_array / 256, dtype=np.uint8)
        inst_final = np.array(seg_inst_array % 256, dtype=np.uint8)

        # Pack data
        data = {
            'datastream_1': {  # image
                'name': '{}.jpg'.format(ts_stop),
                'image': rgb_path
            },
            'datastream_2': {  # sem_seg
                'name': '{}.png'.format(ts_stop),
                'semantic': semseg_final,
                'instances': inst_final
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
    images_names = os.listdir('{}/training/image_2'.format(args.path))
    total_images = len(images_names)

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    # View data
    for idx, img_name in enumerate(images_names):
        # Read images
        rgb_img = cv2.imread('{}/training/image_2/{}'.format(args.path, img_name))
        seg_img = cv2.imread('{}/training/semantic_rgb/{}'.format(args.path, img_name))

        final_img = cv2.vconcat([rgb_img, seg_img])
        # final_img = cv2.resize(final_img, (1440, 600))

        # Show final image
        cv2.imshow('Images', final_img)
        cv2.setWindowTitle("Images", '{}/{} : {}'.format(idx+1, total_images, img_name))
        cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(description='Convert Kitti Semantics Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default='C:/dev/Databases/Kitti/Semantics',
                        help='Path to the original dataset')
    parser.add_argument('--output_path', '-o', default='C:/dev/Databases/Kitti/Semantics/converted',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--limit', default=-1, type=int,
                        help='Number of images to be converted. Set to -1 to convert all db.')

    parser.add_argument('--mode', '-m', default='convert', help='view / convert')

    args = parser.parse_args()

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(args.mode))


if __name__ == '__main__':
    main()
