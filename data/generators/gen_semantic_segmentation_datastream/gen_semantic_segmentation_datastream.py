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
Input: CoreID and FilterID of state measurement datastream
Output: Semantic segmentation datastream (given CoreID and FilterID of the semantic segmentation datastream)
'''

import cv2
import numpy as np
import data.generators.gen_semantic_segmentation_datastream.camera_model
import os
import csv

import sys
sys.path.insert(1, '../../')
from data.CyC_db_interface import CyC_FilterType, CyC_DataType, CyC_DatabaseParser

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

# set input for creating labels
core_id_input = 1
filter_id_input = 6  # must be CyC_STATE_MEASUREMENT

core_id_output = 1
filter_id_output = 72  # a new datastream id

# Set seg filter inputs for blockchain descriptor
core_id_input_img = 1 # CameraFront datastream
filter_id_input_img = 40

core_id_RoadDrivingDNN = 1 # RoadDrivingDNN datastream
filter_id_RoadDrivingDNN = 30

db_path = 'D:/pionner/park/run_4'

buffer_size = 100
cam_param_file = 'usb_camera_logitech.cal'
cam_pose_file = 'pionner_camera_extrinsics.cal'


def world2cam(cam_model, point_3d):
    # if point_3d[0] = 0 it will generate infinity => not OK
    return np.array([cam_model.fx * (point_3d[0][0] / point_3d[2][0]) + cam_model.cx,
                     cam_model.fy * (point_3d[1][0] / point_3d[2][0]) + cam_model.cy])


def create_y_rotation(theta):
    t = np.array([[np.cos(theta),  0, np.sin(theta), 0],
                  [0,              1, 0,             0],
                  [-np.sin(theta), 0, np.cos(theta), 0],
                  [0,              0, 0,             1]])
    return t


def create_z_rotation(theta):
    t = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                  [np.sin(theta),  np.cos(theta), 0, 0],
                  [0,              0,             1, 0],
                  [0,              0,             0, 1]])
    return t


def translate_2d(point, target):
    x = point[0] - target[0]
    y = point[1] - target[1]
    return np.array([x, y])


def rotate_2d(point, angle):
    p = np.array([point])
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    new_p = np.dot(rot_mat, p.T)
    return np.array([new_p[0][0], new_p[1][0]])


def from_2d_to_3d(points):
    return [np.array([[point[0], point[1], 0, 1]]).T for point in points]


def translate_3d(point, target):
    x = point[0][0] - target[0][0]
    y = point[1][0] - target[1][0]
    z = point[2][0] - target[2][0]
    return np.array([[x, y, z, 1]]).T


def rotate_3d(point, mat):
    p_new = np.dot(mat, point)
    return p_new



points = []
angles = []
velos = []

ts_stop = []
ts_start = []
sample_time = []
csv_header = 'timestamp_start,timestamp_stop,sampling_time,left_file_path_0,right_file_path_0'


def main1():
    parser = CyC_DatabaseParser(db_path)

    if parser.get_data_type_by_id(core_id_input, filter_id_input) != CyC_DataType.CyC_STATE_MEASUREMENT:
        print("Input data type not valid, expected: CyC_STATE_MEASUREMENT")
        exit()

    if parser.get_data_type_by_id(core_id_output, filter_id_output) is not None:
        print("Output datastream already exists, please use another id")
        exit()

    datastream_name = 'datastream_{}_{}'.format(core_id_output, filter_id_output)
    photos_folder = os.path.join(db_path, datastream_name, 'samples',
                                 'left')
    # Make dirs
    try:
        os.makedirs(photos_folder)
    except OSError:
        pass

    # Make data_descriptor
    data_csv_file = open(os.path.join(db_path, datastream_name, 'data_descriptor.csv'), 'w')
    data_csv_file.write(csv_header + '\n')

    # Modify Blockchain csv
    blockchain_csv_file_path = os.path.join(db_path, 'datablock_descriptor.csv')
    blockchain_csv_file_path_new = os.path.join(db_path, 'datablock_descriptor_new.csv')

    blockchain_csv_file = open(blockchain_csv_file_path, 'r')
    blockchain_csv_file_new = open(blockchain_csv_file_path_new, 'w', newline='')

    blockchain_csv_reader = csv.reader(blockchain_csv_file, delimiter=',')
    blockchain_csv_writer = csv.writer(blockchain_csv_file_new, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')

    for row in blockchain_csv_reader:
        blockchain_csv_writer.writerow(row)
    _inputs = '{{{0}-{1};{2}-{3}}}'.format(core_id_input_img,
                                           filter_id_input_img,
                                           core_id_RoadDrivingDNN,
                                           filter_id_RoadDrivingDNN)
    blockchain_csv_writer.writerow([core_id_output, filter_id_output, 'SemanticSegCamFront', 24, 4, _inputs])
    blockchain_csv_file.close()
    blockchain_csv_file_new.close()


    # Open Sync csv and write head
    sync_csv_file_path = os.path.join(db_path, 'sampling_timestamps_sync.csv')
    sync_csv_file_path_new = os.path.join(db_path, 'sampling_timestamps_sync_new.csv')

    sync_csv_file = open(sync_csv_file_path, 'r')
    sync_csv_file_new = open(sync_csv_file_path_new, 'w', newline='')

    sync_csv_reader = csv.reader(sync_csv_file, delimiter=',')
    sync_csv_writer = csv.writer(sync_csv_file_new, delimiter=',')

    sync_head = next(sync_csv_reader)
    sync_csv_writer.writerow(sync_head + [datastream_name])



    #
    generator = parser.get_data_by_id(core_id_input, filter_id_input)

    # Load cam params
    cam_itx = camera_model.Camera(cam_param_file)
    cam_ext = camera_model.CameraPose(cam_pose_file)

    cam_pos_3d = np.array([[cam_ext.x, cam_ext.y, cam_ext.z]]).T  # Get cam extrensic params

    rot_mats = list()
    rot_mats.append(create_y_rotation(np.deg2rad(90)))
    rot_mats.append(create_z_rotation(np.deg2rad(-90)))
    rot_mats.append(create_y_rotation(np.deg2rad(90)))
    rot_mats.append(create_y_rotation(np.deg2rad(89)))

    idx = 0
    while idx < buffer_size:
        try:
            _ts_start, _ts_stop, _sample_time, _state = next(generator)
            idx += 1

            points.append(np.array([float(_state[0]), float(_state[1])]))
            angles.append(float(_state[3]))
            velos.append(float(_state[2]))

            ts_start.append(int(_ts_start))
            ts_stop.append(int(_ts_stop))
            sample_time.append(int(_sample_time))
        except StopIteration:
            break

    track_width = 0.75  # half the value
    while True:
        points_of_contact = []
        for p, a in zip(points, angles):
            ang_off = np.deg2rad(90)
            points_of_contact.append(np.array([p[0] + track_width * np.cos(a + ang_off),
                                               p[1] + track_width * np.sin(a + ang_off)]))

            points_of_contact.append(np.array([p[0] + track_width * np.cos(a - ang_off),
                                               p[1] + track_width * np.sin(a - ang_off)]))

        points2D = [translate_2d(p, points[0]) for p in points_of_contact]  # Translate from world to car
        points2D = [rotate_2d(p, -angles[0]) for p in points2D]  # Rotate from world to car

        points3D = from_2d_to_3d(points2D)

        points3D = [translate_3d(p, cam_pos_3d) for p in points3D]

        # Rotate to cam axis
        for rot_mat in rot_mats:
            points3D = [rotate_3d(point, rot_mat) for point in points3D]

        tresh = velos[1] * 2.5
        tresh = 35

        points3D = [world2cam(cam_itx, point) for point in points3D if point[2][0] > 0.5 and point[2][0] < tresh]

        image = np.zeros((int(cam_itx.image_height), int(cam_itx.image_width), 3), dtype='uint8')
        for j in range(0, len(points3D)):
            pts = points3D[j:j + 3]
            pts = np.asarray(pts)
            cv2.fillConvexPoly(image, pts.astype('int32'), (0, 255, 0))

        cv2.imwrite(os.path.join(photos_folder, '{}.png'.format(ts_stop[0])), image)

        #cv2.imshow('img', image)
        #cv2.waitKey(30)

        data_csv_file.write('{},{},{},samples/left/{}.png,-1\n'.format(ts_start[0], ts_stop[0], sample_time[0], ts_stop[0]))

        # Find current Ts_Stop in sync file and write in the new col
        while True:
            line = next(sync_csv_reader)
            if int(line[1]) == int(ts_stop[0]):
                sync_csv_writer.writerow(line + [ts_stop[0]])
                break
            else:
                sync_csv_writer.writerow(line + [-1])

        # Get next sample
        try:
            _ts_start, _ts_stop, _sample_time, _state = next(generator)

            points.pop(0)
            angles.pop(0)
            velos.pop(0)

            ts_start.pop(0)
            ts_stop.pop(0)
            sample_time.pop(0)

            ts_start.append(int(_ts_start))
            ts_stop.append(int(_ts_stop))
            sample_time.append(int(_sample_time))

            points.append(np.array([float(_state[0]), float(_state[1])]))
            angles.append(float(_state[3]))
            velos.append(float(_state[2]))
        except StopIteration:
            # Fill the rest of the sync file
            while True:
                try:
                    line = next(sync_csv_reader)
                    sync_csv_writer.writerow(line + [-1])
                except StopIteration:
                    break

            # Clean up and replace file
            data_csv_file.close()
            sync_csv_file.close()
            sync_csv_file_new.close()

            os.remove(sync_csv_file_path)
            os.rename(sync_csv_file_path_new,
                      sync_csv_file_path)

            os.remove(blockchain_csv_file_path)
            os.rename(blockchain_csv_file_path_new,
                      blockchain_csv_file_path)
            break


def main2():
    parser = CyC_DatabaseParser(db_path)

    if parser.get_data_type_by_id(core_id_input, filter_id_input) != CyC_DataType.CyC_STATE_MEASUREMENT:
        print("Input data type not valid, expected: CyC_STATE_MEASUREMENT")
        exit()

    if parser.get_data_type_by_id(core_id_output, filter_id_output) is not None:
        print("Output datastream already exists, please use another id")
        exit()

    datastream_name = 'datastream_{}_{}'.format(core_id_output, filter_id_output)
    photos_folder = os.path.join(db_path, datastream_name, 'samples',
                                 'left')
    # Make dirs
    try:
        os.makedirs(photos_folder)
    except OSError:
        pass

    # Make data_descriptor
    data_csv_file = open(os.path.join(db_path, datastream_name, 'data_descriptor.csv'), 'w')
    data_csv_file.write(csv_header + '\n')

    # Modify Blockchain csv
    blockchain_csv_file_path = os.path.join(db_path, 'datablock_descriptor.csv')
    blockchain_csv_file_path_new = os.path.join(db_path, 'datablock_descriptor_new.csv')

    blockchain_csv_file = open(blockchain_csv_file_path, 'r')
    blockchain_csv_file_new = open(blockchain_csv_file_path_new, 'w', newline='')

    blockchain_csv_reader = csv.reader(blockchain_csv_file, delimiter=',')
    blockchain_csv_writer = csv.writer(blockchain_csv_file_new, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')

    for row in blockchain_csv_reader:
        blockchain_csv_writer.writerow(row)
    _inputs = '{{{0}-{1};{2}-{3}}}'.format(core_id_input_img,
                                           filter_id_input_img,
                                           core_id_RoadDrivingDNN,
                                           filter_id_RoadDrivingDNN)
    blockchain_csv_writer.writerow([core_id_output, filter_id_output, 'SemanticSegCamFront', 24, 4, _inputs])
    blockchain_csv_file.close()
    blockchain_csv_file_new.close()


    # Open Sync csv and write head
    sync_csv_file_path = os.path.join(db_path, 'sampling_timestamps_sync.csv')
    sync_csv_file_path_new = os.path.join(db_path, 'sampling_timestamps_sync_new.csv')

    sync_csv_file = open(sync_csv_file_path, 'r')
    sync_csv_file_new = open(sync_csv_file_path_new, 'w', newline='')

    sync_csv_reader = csv.reader(sync_csv_file, delimiter=',')
    sync_csv_writer = csv.writer(sync_csv_file_new, delimiter=',')

    sync_head = next(sync_csv_reader)
    sync_csv_writer.writerow(sync_head + [datastream_name])

    #
    generator = parser.get_data_by_id(core_id_input, filter_id_input)

    # Load cam params
    cam_itx = camera_model.Camera(cam_param_file)
    cam_ext = camera_model.CameraPose(cam_pose_file)

    cam_pos_3d = np.array([[cam_ext.x, cam_ext.y, cam_ext.z]]).T  # Get cam extrensic params

    rot_mats = list()
    rot_mats.append(create_y_rotation(np.deg2rad(90)))
    rot_mats.append(create_z_rotation(np.deg2rad(-90)))
    rot_mats.append(create_y_rotation(np.deg2rad(90)))
    rot_mats.append(create_y_rotation(np.deg2rad(89)))

    while True:
        try:
            _ts_start, _ts_stop, _sample_time, _state = next(generator)

            points.append(np.array([float(_state[0]), float(_state[1])]))
            angles.append(float(_state[3]))
            velos.append(float(_state[2]))

            ts_start.append(int(_ts_start))
            ts_stop.append(int(_ts_stop))
            sample_time.append(int(_sample_time))
        except StopIteration:
            break

    track_width = 0.75  # half the value

    points_of_contact = []
    for p, a in zip(points, angles):
        ang_off = np.deg2rad(90)
        points_of_contact.append(np.array([p[0] + track_width * np.cos(a + ang_off),
                                           p[1] + track_width * np.sin(a + ang_off)]))

        points_of_contact.append(np.array([p[0] + track_width * np.cos(a - ang_off),
                                           p[1] + track_width * np.sin(a - ang_off)]))

    idx = 0
    last_idx = len(points) - 50
    while idx < last_idx:
        points2D = [translate_2d(p, points[idx]) for p in points_of_contact]  # Translate from world to car
        points2D = [rotate_2d(p, -angles[idx]) for p in points2D]  # Rotate from world to car

        points3D = from_2d_to_3d(points2D)

        points3D = [translate_3d(p, cam_pos_3d) for p in points3D]

        # Rotate to cam axis
        for rot_mat in rot_mats:
            points3D = [rotate_3d(point, rot_mat) for point in points3D]

        tresh = velos[1] * 2.5
        tresh = 35

        points3D = [world2cam(cam_itx, point) for point in points3D if point[2][0] > 0.5 and point[2][0] < tresh]

        image = np.zeros((int(cam_itx.image_height), int(cam_itx.image_width), 3), dtype='uint8')
        for j in range(0, len(points3D)):
            pts = points3D[j:j + 3]
            pts = np.asarray(pts)
            cv2.fillConvexPoly(image, pts.astype('int32'), (0, 255, 0))

        cv2.imwrite(os.path.join(photos_folder, '{}.png'.format(ts_stop[idx])), image)

        #cv2.imshow('img', image)
        #cv2.waitKey(30)

        data_csv_file.write('{},{},{},samples/left/{}.png,-1\n'.format(ts_start[idx], ts_stop[idx], sample_time[idx], ts_stop[idx]))

        # Find current Ts_Stop in sync file and write in the new col
        while True:
            line = next(sync_csv_reader)
            if int(line[1]) == int(ts_stop[idx]):
                sync_csv_writer.writerow(line + [ts_stop[idx]])
                break
            else:
                sync_csv_writer.writerow(line + [-1])

        idx = idx + 1

    while True:
        try:
            line = next(sync_csv_reader)
            sync_csv_writer.writerow(line + [-1])
        except StopIteration:
            break

    # Clean up and replace file
    data_csv_file.close()
    sync_csv_file.close()
    sync_csv_file_new.close()

    os.remove(sync_csv_file_path)
    os.rename(sync_csv_file_path_new, sync_csv_file_path)

    os.remove(blockchain_csv_file_path)
    os.rename(blockchain_csv_file_path_new, blockchain_csv_file_path)


def main3():
    parser = CyC_DatabaseParser(db_path)

    if parser.get_data_type_by_id(core_id_input, filter_id_input) != CyC_DataType.CyC_STATE_MEASUREMENT:
        print("Input data type not valid, expected: CyC_STATE_MEASUREMENT")
        exit()

    if parser.get_data_type_by_id(core_id_output, filter_id_output) is not None:
        print("Output datastream already exists, please use another id")
        exit()

    datastream_name = 'datastream_{}_{}'.format(core_id_output, filter_id_output)
    photos_folder = os.path.join(db_path, datastream_name, 'samples', 'left')
    # Make dirs
    try:
        os.makedirs(photos_folder)
    except OSError:
        pass

    # Make data_descriptor
    data_csv_file = open(os.path.join(db_path, datastream_name, 'data_descriptor.csv'), 'w')
    data_csv_file.write(csv_header + '\n')

    # Modify Blockchain csv
    blockchain_csv_file_path = os.path.join(db_path, 'datablock_descriptor.csv')
    blockchain_csv_file_path_new = os.path.join(db_path, 'datablock_descriptor_new.csv')

    blockchain_csv_file = open(blockchain_csv_file_path, 'r')
    blockchain_csv_file_new = open(blockchain_csv_file_path_new, 'w', newline='')

    blockchain_csv_reader = csv.reader(blockchain_csv_file, delimiter=',')
    blockchain_csv_writer = csv.writer(blockchain_csv_file_new, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')

    for row in blockchain_csv_reader:
        blockchain_csv_writer.writerow(row)
    _inputs = '{{{0}-{1};{2}-{3}}}'.format(core_id_input_img,
                                           filter_id_input_img,
                                           core_id_RoadDrivingDNN,
                                           filter_id_RoadDrivingDNN)
    blockchain_csv_writer.writerow([core_id_output, filter_id_output, 'SemanticSegCamFront', 24, 4, _inputs])
    blockchain_csv_file.close()
    blockchain_csv_file_new.close()

    # Open Sync csv and write head
    sync_csv_file_path = os.path.join(db_path, 'sampling_timestamps_sync.csv')
    sync_csv_file_path_new = os.path.join(db_path, 'sampling_timestamps_sync_new.csv')

    sync_csv_file = open(sync_csv_file_path, 'r')
    sync_csv_file_new = open(sync_csv_file_path_new, 'w', newline='')

    sync_csv_reader = csv.reader(sync_csv_file, delimiter=',')
    sync_csv_writer = csv.writer(sync_csv_file_new, delimiter=',')

    sync_head = next(sync_csv_reader)
    sync_csv_writer.writerow(sync_head + [datastream_name])
    image_stream_name = "datastream_{}_{}".format(core_id_input, filter_id_input_img)
    try:
        image_column = sync_head.index(image_stream_name)
    except ValueError:
        print("Did not find image stream datastream_{}_{}".format(core_id_input, filter_id_input_img))
        exit()

    #
    generator = parser.get_data_by_id(core_id_input, filter_id_input)

    # Load cam params
    cam_itx = camera_model.Camera(cam_param_file)
    cam_ext = camera_model.CameraPose(cam_pose_file)

    cam_pos_3d = np.array([[cam_ext.x, cam_ext.y, cam_ext.z]]).T  # Get cam extrensic params

    rot_mats = list()
    rot_mats.append(create_y_rotation(np.deg2rad(90)))
    rot_mats.append(create_z_rotation(np.deg2rad(-90)))
    rot_mats.append(create_y_rotation(np.deg2rad(90))) # Y rot
    rot_mats.append(create_y_rotation(np.deg2rad(90)))

    while True:
        try:
            _ts_start, _ts_stop, _sample_time, _state = next(generator)

            points.append(np.array([float(_state[0]), float(_state[1])]))
            angles.append(float(_state[3]))
            velos.append(float(_state[2]))

            ts_start.append(int(_ts_start))
            ts_stop.append(int(_ts_stop))
            sample_time.append(int(_sample_time))
        except StopIteration:
            break

    track_width = 0.15  # half the value

    points_of_contact = []
    for p, a in zip(points, angles):
        ang_off = np.deg2rad(90)
        a = -a
        points_of_contact.append(np.array([p[0] + track_width * np.cos(a + ang_off),
                                           p[1] + track_width * np.sin(a + ang_off)]))

        points_of_contact.append(np.array([p[0] + track_width * np.cos(a - ang_off),
                                           p[1] + track_width * np.sin(a - ang_off)]))

    generator_photos = parser.get_data_by_id(core_id_input, filter_id_input_img)

    ts_start_photos = []
    ts_stop_photos = []
    sample_time_photos = []
    photos = []
    while True:
        try:
            _ts_start, _ts_stop, _sample_time, _photo, _ = next(generator_photos)

            photos.append(_photo)
            ts_start_photos.append(int(_ts_start))
            ts_stop_photos.append(int(_ts_stop))
            sample_time_photos.append(int(_sample_time))
        except StopIteration:
            break


    idx = 0
    while idx < len(ts_stop_photos):
        for i, ts in enumerate(ts_stop):
            if ts >= ts_stop_photos[idx]:
                current_point = points[i]
                current_angle = angles[i]
                break

        points2D = [translate_2d(p, current_point) for p in points_of_contact]  # Translate from world to car
        points2D = [rotate_2d(p, current_angle) for p in points2D]  # Rotate from world to car

        points3D = from_2d_to_3d(points2D)
        points3D = [translate_3d(p, cam_pos_3d) for p in points3D]

        # Rotate to cam axis
        for rot_mat in rot_mats:
            points3D = [rotate_3d(point, rot_mat) for point in points3D]


        tresh = velos[1] * 2.5
        tresh = 35

        points3D = [world2cam(cam_itx, point) for point in points3D if point[2][0] > 0.1 and point[2][0] < tresh]

        #image = np.zeros((int(cam_itx.image_height), int(cam_itx.image_width), 3), dtype='uint8')
        image = photos[idx]
        for j in range(0, len(points3D)):
            pts = points3D[j:j + 3]
            pts = np.asarray(pts)
            cv2.fillConvexPoly(image, pts.astype('int32'), (0, 255, 0))

        cv2.imwrite(os.path.join(photos_folder, '{}.png'.format(ts_stop_photos[idx])), image)

        #cv2.imshow('img', image)
        #cv2.waitKey(30)

        data_csv_file.write(
            '{},{},{},samples/left/{}.png,-1\n'.format(ts_start_photos[idx], ts_stop_photos[idx], sample_time_photos[idx], ts_stop_photos[idx]))




        idx = idx + 1

    while True:
        try:
            line = next(sync_csv_reader)
            sync_csv_writer.writerow(line + [line[image_column]])
        except StopIteration:
            break

if __name__ == '__main__':
    # main1() # for use with nuScenes only
    # main2() # for use with nuScenes only
    main3()
