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
Script for converting Kitti Odometry Dataset to CyberCortex.AI format

Kitti dataset:
    http://www.cvlibs.net/datasets/kitti/eval_odometry.php

Download:
    Download all the datasets (gray, color, velodyne and calib) and
    unzip them all in the same folder.
"""

import argparse
import os
import time
import cv2
import numpy as np
from data.converters.CyC_DatabaseFormat import CyC_DataBase

"""
Dataset Info:

Tree:
    poses
        00.txt
        01.txt
    sequences
        00
            image_0
                000000.png
                000001.png
                ...
            image_1
                000000.png
                000001.png
                ...
            image_2
                000000.png
                000001.png
                ...
            image_3
                000000.png
                000001.png
                ...
            calib.txt
            times.txt
        01
            ...

# Example of calib and pose
    gray left - intrinsics
    7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00
    0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00
    0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
    
    gray right - intrinsics
    7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02
    0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00
    0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
    
    pose[0] - extrinsics
    1.000000e+00 9.043680e-12 2.326809e-11 5.551115e-17 
    9.043683e-12 1.000000e+00 2.392370e-10 3.330669e-16 
    2.326810e-11 2.392370e-10 9.999999e-01 -4.440892e-16
    
    pose[1] - extrinsics
    9.999978e-01 5.272628e-04 -2.066935e-03 -4.690294e-02 
    -5.296506e-04 9.999992e-01 -1.154865e-03 -2.839928e-02 
    2.066324e-03 1.155958e-03 9.999971e-01 8.586941e-01
"""


def read_velo(velo_path):
    scan = np.fromfile(velo_path, dtype=np.float32)
    return scan.reshape((-1, 4))


def read_calib(calib_path):
    data = {}

    # Load the calibration file
    filedata = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                filedata[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P0'], (3, 4))
    P_rect_10 = np.reshape(filedata['P1'], (3, 4))
    P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    P_rect_30 = np.reshape(filedata['P3'], (3, 4))

    data['P_rect_00'] = P_rect_00
    data['P_rect_10'] = P_rect_10
    data['P_rect_20'] = P_rect_20
    data['P_rect_30'] = P_rect_30

    # Compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
    data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
    data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
    data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
    data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    # Compute the stereo baselines in meters by projecting the origin of
    # each camera frame into the velodyne frame and computing the distances
    # between them
    p_cam = np.array([0, 0, 0, 1])
    p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
    p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
    p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
    p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

    data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
    data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

    return data


def convert_data(args):
    # Count images
    if args.sequence == '-1':
        seqs = ','.join(os.listdir(os.path.join(args.path, 'sequences')))
    else:
        seqs = args.sequence.split(',')

    total_images = 0
    for seq in seqs:
        with open('{}/sequences/{}/times.txt'.format(args.path, seq), 'r') as times_file:
            total_images += len(times_file.readlines())

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    # Create empty database
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='image', filter_id=1, filter_name='CameraFront')
    db.add_stream(filter_type='image', filter_id=2, filter_name='CameraFrontGray')
    db.add_stream(filter_type='lidar', filter_id=3, filter_name='Velodyne')
    # db.show_packing_info()
    db.create_db()

    # Receive data
    for seq in seqs:
        with open('{}/sequences/{}/times.txt'.format(args.path, seq), 'r') as times_file:
            times = times_file.readlines()
        ts_stop = int(float(times[0].strip()))

        # Read calib
        # calib = read_calib('{}/sequences/{}/calib.txt'.format(args.path, seq))

        for idx, ts in enumerate(times):
            # Prepare timestamps
            ts_start = ts_stop
            ts_stop = int(float(ts.strip()) * 1000)

            # Pack data
            data = {
                'datastream_1': {  # image
                    'name': '{}.png'.format(ts_stop),
                    'image': '{}/sequences/{}/image_2/{:06d}.png'.format(args.path, seq, idx),
                    'image_right': '{}/sequences/{}/image_3/{:06d}.png'.format(args.path, seq, idx)
                },
                'datastream_2': {  # image
                    'name': '{}.png'.format(ts_stop),
                    'image': '{}/sequences/{}/image_0/{:06d}.png'.format(args.path, seq, idx),
                    'image_right': '{}/sequences/{}/image_1/{:06d}.png'.format(args.path, seq, idx)
                },
                'datastream_3': {  # lidar
                    'name': '{}.bin'.format(ts_stop),
                    'lidar_file': '{}/sequences/{}/velodyne/{:06d}.bin'.format(args.path, seq, idx)
                }
            }

            # Add data to database
            db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

            # Print progress
            if idx % 25 == 0:
                print('Done {0} out of {1}.'.format(idx, total_images))
        break  # Not functional yet on multiple sequences
    print('Finished process.')


def view_data(args):
    # Count images
    if args.sequence == '-1':
        seqs = ','.join(os.listdir(os.path.join(args.path, 'sequences')))
    else:
        seqs = args.sequence.split(',')

    total_images = 0
    for seq in seqs:
        with open('{}/sequences/{}/times.txt'.format(args.path, seq), 'r') as times_file:
            total_images += len(times_file.readlines())

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    # View data
    for seq in seqs:
        with open('{}/sequences/{}/times.txt'.format(args.path, seq), 'r') as times_file:
            times = times_file.readlines()

        # Read calib
        # calib = read_calib('{}/sequences/{}/calib.txt'.format(args.path, seq))

        for idx, ts in enumerate(times):
            # Read images
            gray_l = cv2.imread('{}/sequences/{}/image_0/{:06d}.png'.format(args.path, seq, idx))
            gray_r = cv2.imread('{}/sequences/{}/image_1/{:06d}.png'.format(args.path, seq, idx))
            rgb_l = cv2.imread('{}/sequences/{}/image_2/{:06d}.png'.format(args.path, seq, idx))
            rgb_r = cv2.imread('{}/sequences/{}/image_3/{:06d}.png'.format(args.path, seq, idx))

            # Read velodyne - [x,y,z,reflectance]
            # velo = read_velo('{}/sequences/{}/velodyne/{:06d}.bin'.format(args.path, seq, idx))

            final_img = cv2.vconcat([
                cv2.hconcat([gray_l, gray_r]),
                cv2.hconcat([rgb_l, rgb_r]),

            ])
            final_img = cv2.resize(final_img, (1440, 600))

            # Show final image
            cv2.imshow('Images', final_img)
            cv2.setWindowTitle("Images", 'Seq {}, {}/{}'.format(seq, idx, total_images))
            cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(description='Convert Kitti Odometry Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default='C:/dev/Databases/Kitti/Odometry/dataset',
                        help='Path to the original dataset')
    parser.add_argument('--output_path', '-o', default='C:/dev/Databases/Kitti/Odometry/converted',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--sequence', '-s', default='00',
                        help='Sequences to be converted. Example: 00,01,02. Set to -1 to convert all.')

    parser.add_argument('--mode', '-m', default='convert', help='view / convert')

    if parser.parse_args().mode == 'convert':
        convert_data(args=parser.parse_args())
    elif parser.parse_args().mode == 'view':
        view_data(args=parser.parse_args())
    else:
        print('Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(parser.parse_args().mode))


if __name__ == '__main__':
    main()
