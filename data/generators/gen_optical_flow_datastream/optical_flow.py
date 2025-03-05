"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

'''
Script for generating optical flow datastream for an [IMAGE] datastream from fake_dataset.
'''

import cv2
import numpy as np
import os
import csv
import sys
import pandas as pd

sys.path.insert(1, '../../')
import global_config
from CyC_db_interface import CyC_FilterType, CyC_DataType, CyC_DatabaseParser

# DEFINE CONSTANTS FOR OPTICAL FLOW ESTIMATION METHODS
LUCAS_KANADE = False
FARNEBACK = True

# Image size parameters
res_W = 800
res_H = 600
# DEFINE FLAGS TO WAIT as CV delay
CV_DELAY = 1
# Get global config
CFG = global_config.cfg

# set input for computing optical flow
core_id_input = 1
filter_id_input = 1

# new datastream id
core_id_output = 1
filter_id_output = 99

# Set corresponding datastream for blockchain descriptor
core_id_input_img = 1
filter_id_input_img = 1

# Define csv file headers
ts_stop = []
ts_start = []
sample_time = []
saved_images_paths = []
csv_header = 'timestamp_start,timestamp_stop,sampling_time,left_file_path_0,right_file_path_0'


# Implementation
def main():
    database_path = CFG.DB.BASE_PATH
    parser = CyC_DatabaseParser(database_path)

    if parser.get_data_type_by_id(core_id_input, filter_id_input) != CyC_DataType.CyC_IMAGE:
        print("Input data type not valid, expected: CyC_STATE_MEASUREMENT")
        exit()

    if parser.get_data_type_by_id(core_id_output, filter_id_output) is not None:
        print("Output datastream already exists, please use another id")
        exit()

    datastream_name = 'datastream_{}_{}'.format(core_id_output, filter_id_output)
    photos_folder = os.path.join(database_path, datastream_name, 'samples', 'left')

    # Make dirs
    try:
        os.makedirs(photos_folder)
    except OSError:
        pass


    # Modify Blockchain csv
    blockchain_csv_file_path = os.path.join(database_path, 'datablock_descriptor.csv')
    blockchain_csv_file_path_new = os.path.join(database_path, 'datablock_descriptor_new.csv')

    blockchain_csv_file = open(blockchain_csv_file_path, 'r')
    blockchain_csv_file_new = open(blockchain_csv_file_path_new, 'w', newline='')

    blockchain_csv_reader = csv.reader(blockchain_csv_file, delimiter=',')
    blockchain_csv_writer = csv.writer(blockchain_csv_file_new, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')

    for row in blockchain_csv_reader:
        blockchain_csv_writer.writerow(row)
    _inputs = '{{{0}-{1}}}'.format(core_id_input_img, filter_id_input_img)
    blockchain_csv_writer.writerow([core_id_output, filter_id_output, 'OpticalFlowCamFront', 1, 4, _inputs]) # 1, 1 core type, filter type = image, image
    blockchain_csv_file.close()
    blockchain_csv_file_new.close()


    # Get image datastream generator (1, 1) CAM FRONT
    generator = parser.get_data_by_id(core_id_input, filter_id_input)

    if LUCAS_KANADE:
        # Setup parameters for computing Optical Flow using Lucas-Kanade method from OpenCV
        # For reference see: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))

        # Get first frame for reference
        _ts_start, _ts_stop, _sample_time, _first_frame, _ = next(generator)

        # Pre-process first frame
        old_gray = cv2.cvtColor(_first_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(_first_frame)
    elif FARNEBACK:
        # Get first frame for reference
        _ts_start, _ts_stop, _sample_time, _first_frame, _ = next(generator)

    #_first_frame = cv2.resize(_first_frame, (res_W, res_H))
    do_break = False
    while not do_break:
        try:
            ts_start.append(int(_ts_start))
            ts_stop.append(int(_ts_stop))
            sample_time.append(int(_sample_time))
            if LUCAS_KANADE:
                # Get next frame
                _ts_start, _ts_stop, _sample_time, _next_frame, _ = next(generator)
                _next_frame = cv2.cvtColor(_next_frame, cv2.COLOR_BGR2RGB)
                #_next_frame = cv2.resize(_next_frame, (res_W, res_H))
                frame_gray = cv2.cvtColor(_next_frame, cv2.COLOR_BGR2GRAY)
                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    _next_frame = cv2.circle(_next_frame, (a, b), 5, color[i].tolist(), -1)
                img = cv2.add(_next_frame, mask)
                img = cv2.hconcat(_next_frame, img)
                cv2.imshow('frame', img)
                cv2.waitKey(CV_DELAY)

                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            elif FARNEBACK:
                # Compute optical flow
                first_frame = _first_frame
                prvs = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                hsv = np.zeros_like(first_frame)
                hsv[..., 1] = 255

                # Get next frame
                _ts_start, _ts_stop, _sample_time, _next_frame, _ = next(generator)
                _next_frame = cv2.cvtColor(_next_frame, cv2.COLOR_BGR2RGB)
                #_next_frame = cv2.resize(_next_frame,(res_W, res_H))
                next_frame = cv2.cvtColor(_next_frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 5, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                im_h = cv2.hconcat([_next_frame, bgr])
                # Disable temporarely
                save_img_path = os.path.join(photos_folder, '{}.png'.format(int(_ts_stop)))
                cv2.imwrite(save_img_path, bgr)
                saved_images_paths.append(save_img_path)
                # show image for testing
                cv2.imshow("OPTICAL", im_h)
                cv2.waitKey(CV_DELAY)
        except StopIteration:
            do_break = True
        if do_break:
            break

    # Define sync files paths. Will have to declare 2 files, reader and writer as writing and reading at the same time, is a bad idea.
    sync_csv_file_path = os.path.join(database_path, 'sampling_timestamps_sync.csv')
    sync_csv_file_new_path = os.path.join(database_path, 'sampling_timestamps_sync_new.csv')

    sync_csv_file = open(sync_csv_file_path, 'r')
    sync_csv_file_new = open(sync_csv_file_new_path, 'w', newline='')

    sync_csv_reader = csv.reader(sync_csv_file, delimiter=',')
    sync_csv_writer = csv.writer(sync_csv_file_new, delimiter=',')

    sync_head = next(sync_csv_reader)
    sync_csv_writer.writerow(sync_head + [datastream_name])

    # Make data_descriptor
    with open(os.path.join(database_path, datastream_name, 'data_descriptor.csv'), 'w') as data_csv_file:
        data_csv_file.write(csv_header + '\n')
        for _ts_start, _ts_stop, _s_time, _im_path in zip(ts_start, ts_stop, sample_time, saved_images_paths):
            im_path = _im_path.split('\\')
            data_csv_file.write('{},{},{},{},-1\n'.format(_ts_start, _ts_stop, _sample_time, '/'.join(im_path[-3:])))
        data_csv_file.close()

    while True:
        try:
            row = next(sync_csv_reader)
            _t_stop = row[1]
            if int(_t_stop) in ts_stop:
                sync_csv_writer.writerow(row + [_t_stop])
            else:
                sync_csv_writer.writerow(row + [-1])
        except StopIteration:
            break

    # Replace old files with newly written ones.
    sync_csv_file.close()
    sync_csv_file_new.close()
    os.remove(sync_csv_file_path)
    os.remove(blockchain_csv_file_path)
    os.rename(sync_csv_file_new_path, sync_csv_file_path)
    os.rename(blockchain_csv_file_path_new, blockchain_csv_file_path)


if __name__ == '__main__':
    main()
