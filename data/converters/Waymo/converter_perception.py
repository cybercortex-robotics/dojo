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
Script for converting Waymo Perception dataset to CyberCortex.AI

Dataset link:
    https://waymo.com/intl/en_us/open/download/ - Perception Dataset

Download commands:
    - testing:           <Waymo_path>$ gsutil -m cp -r "gs://waymo_open_dataset_v_1_4_2/individual_files/testing" .
    - training:          <Waymo_path>$ gsutil -m cp -r "gs://waymo_open_dataset_v_1_4_2/individual_files/training" .
    - validation:        <Waymo_path>$ gsutil -m cp -r "gs://waymo_open_dataset_v_1_4_2/individual_files/validation" .

Gsutil downloads:
    https://cloud.google.com/storage/docs/gsutil_install#windows - gsutil install tutorial
    https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe - gsutil installer

Latest verified database version for converter: v1.4.2

Inspiration:
    https://salzi.blog/2022/05/14/waymo-open-dataset-open3d-point-cloud-viewer/
"""

import argparse
import os
import time
import cv2
import glob
import numpy as np

import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.label_pb2 import Label
from waymo_open_dataset.utils import frame_utils, transform_utils

from data.converters.Waymo.utils import *
from data.converters.CyC_DatabaseFormat import CyC_DataBase, mkdir

"""
Space required:
    - testing:         143.7 GiB
    - training:        760.3 GiB
    - validation:      191.5 GiB

Tree:
    testing
        segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord
        <>.tfrecord
    training
        segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
        <>.tfrecord
    validation
        segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord
        <>.tfrecord
        
Tfrecord format:
    Check "frame" parameters.
    
Calib examples:
 - Intrinsic mat:
    [[2059.0471439559833, 2059.0471439559833, 935.1248081874216], 
     [635.052474560227, 0.04239636827428756, -0.34165672675852826], 
     [0.001805535524580487, -5.530628187935031e-05, 0.0]]
 - Extrinsic mat:
    [[0.9999944135207451, 0.0017267926275759344, -0.002862012320402869, 1.5439641908208435, 
     [-0.0016833005658143062, 0.9998841227217824, 0.015129693588982756, -0.02326789235447021], 
     [0.002887806521551894, -0.01512479144030509, 0.9998814436008807, 2.1153331179684765], 
     [0.0, 0.0, 0.0, 1.0]]
"""


def convert_data(args):
    # Count data
    tf_files = tf.io.gfile.glob('{}/{}/*.tfrecord'.format(args.path, args.set))
    print('Found {} tfrecords.'.format(len(tf_files)))

    if args.limit > 0:
        print('Converting only {}.'.format(args.limit))

    # Create database folder
    mkdir(args.output_path)

    # Size of database
    size = (args.width, args.height)

    # Receive data
    for tf_idx, tf_file in enumerate(tf_files):
        tf_dataset = tf.data.TFRecordDataset(tf_file, compression_type='')

        print('\nStarted conversion of tfrecord {}.'.format(tf_idx + 1))

        # Create empty database with the index tf_idx+1 for every stream
        db = CyC_DataBase(db_path='{}/DB_{}'.format(args.output_path, tf_idx + 1), core_id=1)
        db.add_stream(filter_type='camera', filter_id=1, filter_name='CamFront')
        db.add_stream(filter_type='camera', filter_id=2, filter_name='CamLeft')
        db.add_stream(filter_type='camera', filter_id=3, filter_name='CamRight')
        db.add_stream(filter_type='camera', filter_id=4, filter_name='CamSideLeft')
        db.add_stream(filter_type='camera', filter_id=5, filter_name='CamSideRight')
        db.add_stream(filter_type='det2d', filter_id=6, filter_name='Det2DFront', input_sources=[1])
        db.add_stream(filter_type='det2d', filter_id=7, filter_name='Det2DLeft', input_sources=[2])
        db.add_stream(filter_type='det2d', filter_id=8, filter_name='Det2DRight', input_sources=[3])
        db.add_stream(filter_type='det2d', filter_id=9, filter_name='Det2DSideLeft', input_sources=[4])
        db.add_stream(filter_type='det2d', filter_id=10, filter_name='Det2DSideRight', input_sources=[5])
        db.add_stream(filter_type='lidar', filter_id=11, filter_name='Lidar')
        db.add_stream(filter_type='det3d', filter_id=12, filter_name='Det3D', input_sources=[1, 2, 3, 4, 5])
        # db.show_packing_info()
        db.create_db()

        # Add object classes
        obj_cls_file = create_obj_cls()
        db.add_custom(filter_id=6, data=obj_cls_file, name='object_classes.conf')
        db.add_custom(filter_id=7, data=obj_cls_file, name='object_classes.conf')
        db.add_custom(filter_id=8, data=obj_cls_file, name='object_classes.conf')
        db.add_custom(filter_id=9, data=obj_cls_file, name='object_classes.conf')
        db.add_custom(filter_id=10, data=obj_cls_file, name='object_classes.conf')
        db.add_custom(filter_id=12, data=obj_cls_file, name='object_classes.conf')

        ts_start = 0
        for data_idx, data in enumerate(tf_dataset):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            ts_stop = frame.timestamp_micros

            if data_idx == 0:  # Make only once
                # Prepare timestamp
                ts_start = ts_stop

                # Get camera calibration TODO Save params, not matrices
                cal_front = cal_left = cal_right = cal_side_left = cal_side_right = ''
                for calib in frame.context.camera_calibrations:
                    if calib.name == dataset_pb2.CameraName.FRONT:
                        cal_front = generate_matrices_calib(calib.extrinsic.transform,
                                                            calib.intrinsic)
                    if calib.name == dataset_pb2.CameraName.FRONT_LEFT:
                        cal_left = generate_matrices_calib(calib.extrinsic.transform,
                                                           calib.intrinsic)
                    if calib.name == dataset_pb2.CameraName.FRONT_RIGHT:
                        cal_right = generate_matrices_calib(calib.extrinsic.transform,
                                                            calib.intrinsic)
                    if calib.name == dataset_pb2.CameraName.SIDE_LEFT:
                        cal_side_left = generate_matrices_calib(calib.extrinsic.transform,
                                                                calib.intrinsic)
                    if calib.name == dataset_pb2.CameraName.SIDE_RIGHT:
                        cal_side_right = generate_matrices_calib(calib.extrinsic.transform,
                                                                 calib.intrinsic)

                # Get lidar calibration TODO Save params, not matrices
                cal_lidar_top = ''
                for calib in frame.context.laser_calibrations:
                    if calib.name == dataset_pb2.LaserName.TOP:
                        cal_lidar_top = generate_matrices_calib(calib.extrinsic.transform, None)
                        break

                # Save calib file
                db.add_custom(filter_id=1, data=cal_front, name='calibration.cal')
                db.add_custom(filter_id=2, data=cal_left, name='calibration.cal')
                db.add_custom(filter_id=3, data=cal_right, name='calibration.cal')
                db.add_custom(filter_id=4, data=cal_side_left, name='calibration.cal')
                db.add_custom(filter_id=5, data=cal_side_right, name='calibration.cal')
                db.add_custom(filter_id=11, data=cal_lidar_top, name='calibration.cal')

            # Process images
            img_front = img_left = img_right = img_side_left = img_side_right = None
            for image in frame.images:
                if image.name == dataset_pb2.CameraName.FRONT:
                    img_front = process_image(image.image, size)
                    # seg_front = tf.image.decode_png(image.camera_segmentation_label.panoptic_label)  # TODO SemSeg?
                elif image.name == dataset_pb2.CameraName.FRONT_LEFT:
                    img_left = process_image(image.image, size)
                elif image.name == dataset_pb2.CameraName.FRONT_RIGHT:
                    img_right = process_image(image.image, size)
                elif image.name == dataset_pb2.CameraName.SIDE_LEFT:
                    img_side_left = process_image(image.image, size)
                elif image.name == dataset_pb2.CameraName.SIDE_RIGHT:
                    img_side_right = process_image(image.image, size)

            # Process labels => Det 2D
            front_det2d = left_det2d = right_det2d = side_left_det2d = side_right_det2d = {}
            for labels in frame.camera_labels:
                if labels.name == dataset_pb2.CameraName.FRONT:
                    front_det2d = generate_det2d(labels.labels)
                elif labels.name == dataset_pb2.CameraName.FRONT_LEFT:
                    left_det2d = generate_det2d(labels.labels)
                elif labels.name == dataset_pb2.CameraName.FRONT_RIGHT:
                    right_det2d = generate_det2d(labels.labels)
                elif labels.name == dataset_pb2.CameraName.SIDE_LEFT:
                    side_left_det2d = generate_det2d(labels.labels)
                elif labels.name == dataset_pb2.CameraName.SIDE_RIGHT:
                    side_right_det2d = generate_det2d(labels.labels)

            # Process labels => Det 2D
            det_3d = generate_det3d(frame.laser_labels)

            # Process Lidar
            lidar_pts = process_lidar(frame)  # [[x, y, z], ...]

            # Pack data
            data = {
                'datastream_1': {  # image
                    'name': '{}.png'.format(ts_stop),
                    'image': img_front
                },
                'datastream_2': {  # image
                    'name': '{}.png'.format(ts_stop),
                    'image': img_left
                },
                'datastream_3': {  # image
                    'name': '{}.png'.format(ts_stop),
                    'image': img_right
                },
                'datastream_4': {  # image
                    'name': '{}.png'.format(ts_stop),
                    'image': img_side_left
                },
                'datastream_5': {  # image
                    'name': '{}.png'.format(ts_stop),
                    'image': img_side_right
                },
                'datastream_6': {  # 2D_det
                    'frame_id': data_idx,
                    'roi_id': front_det2d['id'],
                    'cls': front_det2d['cls'],
                    'x': front_det2d['x'],
                    'y': front_det2d['y'],
                    'width': front_det2d['w'],
                    'height': front_det2d['h']
                },
                'datastream_7': {  # 2D_det
                    'frame_id': data_idx,
                    'roi_id': left_det2d['id'],
                    'cls': left_det2d['cls'],
                    'x': left_det2d['x'],
                    'y': left_det2d['y'],
                    'width': left_det2d['w'],
                    'height': left_det2d['h']
                },
                'datastream_8': {  # 2D_det
                    'frame_id': data_idx,
                    'roi_id': right_det2d['id'],
                    'cls': right_det2d['cls'],
                    'x': right_det2d['x'],
                    'y': right_det2d['y'],
                    'width': right_det2d['w'],
                    'height': right_det2d['h']
                },
                'datastream_9': {  # 2D_det
                    'frame_id': data_idx,
                    'roi_id': side_left_det2d['id'],
                    'cls': side_left_det2d['cls'],
                    'x': side_left_det2d['x'],
                    'y': side_left_det2d['y'],
                    'width': side_left_det2d['w'],
                    'height': side_left_det2d['h']
                },
                'datastream_10': {  # 2D_det
                    'frame_id': data_idx,
                    'roi_id': side_right_det2d['id'],
                    'cls': side_right_det2d['cls'],
                    'x': side_right_det2d['x'],
                    'y': side_right_det2d['y'],
                    'width': side_right_det2d['w'],
                    'height': side_right_det2d['h']
                },
                'datastream_11': {  # lidar
                    'name': '{}.bin.ply'.format(ts_stop),
                    'lidar_data': lidar_pts
                },
                'datastream_12': {  # 3D_det
                    'frame_id': data_idx,
                    'roi_id': det_3d['id'],
                    'cls': det_3d['cls'],
                    'x': det_3d['x'], 'y': det_3d['y'], 'z': det_3d['z'],
                    'w': det_3d['w'], 'h': det_3d['h'], 'l': det_3d['l'],
                    'roll': det_3d['roll'],
                    'pitch': det_3d['pitch'],
                    'yaw': det_3d['yaw']
                }
            }

            # Add data to database
            db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

            # Advance timestamp
            ts_start = ts_stop

            # Display progress
            if data_idx != 0 and data_idx % 5 == 0:
                print('Done {} from tfrecord {}.'.format(data_idx, tf_idx+1))
        print('Finished tfrecord {}.'.format(tf_idx + 1))

        if 0 < args.limit == tf_idx+1:
            break
    print('\nFinished converting the data.')


def view_data(args):
    # Count data
    tf_files = tf.io.gfile.glob('{}/{}/*.tfrecord'.format(args.path, args.set))
    print('Total tfrecords found: {}'.format(len(tf_files)))

    # View data
    for tf_idx, tf_file in enumerate(tf_files):
        tf_dataset = tf.data.TFRecordDataset(tf_file, compression_type='')
        for data in tf_dataset:
            frame = Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # Process Lidar
            lidar_pts = process_lidar(frame)  # [[x, y, z], ...]

            # View Lidar
            show_point_cloud(lidar_pts, frame.laser_labels)

    print(' # Finished viewing all data.')


def main():
    parser = argparse.ArgumentParser(description='Convert Waymo Perception Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default=r'C:\dev\Databases\Waymo',
                        help='Path to the waymo dataset')
    parser.add_argument('--output_path', '-o', default=r'C:\dev\Databases\Waymo_val_converted',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--set', default='validation',
                        help='validation / training')
    parser.add_argument('--width', default=-1, type=int,
                        help='New width of the output image. Set to -1 to keep original size.')
    parser.add_argument('--height', default=-1, type=int,
                        help='New height of the output image. Set to -1 to keep original size.')
    parser.add_argument('--limit', default=-1, type=int,
                        help='Number of tfrecords to be converted. Set to -1 to convert all.')

    parser.add_argument('--mode', '-m', default='convert', help='view / convert')

    args = parser.parse_args()

    # Check arguments
    if args.set not in ['training', 'validation']:
        print('Set is not valid. Try: \'training\' or \'validation\'.')
        exit(1)

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(parser.parse_args().mode))


if __name__ == '__main__':
    main()
