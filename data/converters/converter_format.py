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
Script for converting <> Dataset to CyberCortex.AI
    Author: <>

Dataset link:
    <>
"""

import argparse
import os
import time
import cv2
import glob
import numpy as np

from data.converters.CyC_DatabaseFormat import *

"""
Dataset info:
<>
"""


def convert_data(args):
    # Count data
    pass

    # Create empty database with the index db_idx for every stream
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='<>', filter_id=0, filter_name='<>', input_sources=[0])
    # db.show_packing_info()
    # db.create_db()

    # Prepare timestamp
    ts_start = int(time.time())
    ts_stop = ts_start
    sampling_time = 5

    # Add obj cls file
    obj_cls = create_obj_cls_file()
    db.add_custom(filter_id=0, data=obj_cls, name='object_classes.conf')

    # Add calib file
    calib = create_calib_file()
    db.add_custom(filter_id=0, data=calib, name='calibration.cal')

    # Receive data
    for _ in _:
        # Pack data
        data = {
            # Run show_packing_info() to get the format of data
        }

        # Add data to database
        db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

        # Advance timestamp
        ts_start = ts_stop
        ts_stop += sampling_time
    print(' # Finished converting all data.')


def view_data(args):
    # Count data

    # View data
    for _ in _:
        pass
    print(' # Finished viewing all data.')


def main():
    parser = argparse.ArgumentParser(description='Convert <> Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default=r'C:\Data\<>',
                        help='Path to the waymo dataset')
    parser.add_argument('--output_path', '-o', default=r'C:\Data\<>_converted',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--width', default=800, type=int,
                        help='New width of the output image. Set to -1 to keep original size.')
    parser.add_argument('--height', default=600, type=int,
                        help='New height of the output image. Set to -1 to keep original size.')

    parser.add_argument('--mode', '-m', default='view', help='view / convert')

    args = parser.parse_args()

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(parser.parse_args().mode))


if __name__ == '__main__':
    main()
