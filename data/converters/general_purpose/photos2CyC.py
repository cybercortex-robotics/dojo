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
Script for converting images from a folder to CyberCortex.AI DataBase
"""

import argparse
import os
import cv2
import time
from data.converters.CyC_DatabaseFormat import CyC_DataBase


def convert_data(args):
    images = os.listdir(args.input_path)
    total_images = len(images)

    # Create empty database
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='image', filter_id=1, filter_name='Front_Camera')
    # db.show_packing_info()
    db.create_db()

    # Prepare timestamp
    ts_start = int(time.time())
    ts_stop = ts_start
    sampling_time = 5

    # Receive data
    count = 0
    for image_name in images:
        # Resize if specified
        if args.width != -1 and args.height != -1:
            image = cv2.imread(os.path.join(args.input_path, image_name))
            image = cv2.resize(image, (args.width, args.height))

            # Pack data
            data = {'datastream_1': {  # image
                'name': '{}.jpg'.format(ts_stop),
                'image': image
                }
            }
        else:
            # Pack data
            data = {'datastream_1': {  # image
                'name': '{}.jpg'.format(ts_stop),
                'image': os.path.join(args.input_path, image_name)
                }
            }

        # Add data to database
        db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

        # Advance timestamp
        ts_start = ts_stop
        ts_stop += sampling_time

        # Print progress
        count += 1
        if count % 25 == 0:
            print('Done {0} out of {1}.'.format(count, total_images))

    print('Finished process.')


def view_data(args):
    images = os.listdir(args.input_path)
    # total_images = len(images)

    # Read data
    count = 0
    for image_name in images:
        image = cv2.imread(os.path.join(args.input_path, image_name))

        # Resize if specified
        if args.width != -1 and args.height != -1:
            image = cv2.resize(image, (args.width, args.height))

        # Show image
        cv2.imshow('Image', image)
        cv2.setWindowTitle("Image", 'Image-{0}'.format(count))
        cv2.waitKey(1)

        # Progress
        count += 1


def main():
    parser = argparse.ArgumentParser(description='Useful script for converting images to CyberCortex.AI format')

    parser.add_argument('--input_path', '-i', default=r'C:\data\Architecture\raw\01',
                        help='Path to the image folder')
    parser.add_argument('--output_path', '-o', default=r'C:\data\Architecture\arch_folder_test',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--width', default=int(4000 / 4), type=int,
                        help='Width of images. Set to -1 to not resize.')
    parser.add_argument('--height', default=int(3000 / 4), type=int,
                        help='Height of images. Set to -1 to not resize.')

    parser.add_argument('--mode', '-m', default='convert', help='view / convert')

    args = parser.parse_args()

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode {} does not exist. Try \'view\' or \'convert\'.'.format(args.mode))


if __name__ == '__main__':
    main()
