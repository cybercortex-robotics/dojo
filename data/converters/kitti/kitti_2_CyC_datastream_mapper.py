"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import pykitti
import os
import csv
from datetime import datetime
from shutil import copy
import numpy as np
import math
from utilities.parse_tracklets import load_tracklets_for_frames, parseXML

from toolkit.object_classes import ObjectClasses
from data import types_CyC_TYPES
import global_config

CFG = global_config.cfg

# TODO: fix hardcoded camera intrisics. Talk to add kitti_camera params to CyberCortex.AI pipelines
P2 = np.array([[721.5377, 0., 609.5593, 0.], [0., 721.5377, 172.854, 0.], [0., 0., 1., 0.]])

# Define object classes
object_classes = ObjectClasses(CFG.CyC_INFERENCE.OBJECT_CLASSES_PATH)

# Map kitti classes to object_classes in CyberCortex.AI
kitti_classes_mapper = {'Car': object_classes.object_classes[2],
                        'Van': object_classes.object_classes[3],
                        'Truck': object_classes.object_classes[3],
                        'Pedestrian': object_classes.object_classes[0],
                        'Person_sitting': object_classes.object_classes[0],
                        'Cyclist': object_classes.object_classes[6],
                        'Tram': object_classes.object_classes[3]
                        }

# Change this to the directory where KITTI dataset is
basedir = r'D:/Datasets/KITTI_raw/'

# Change this to the directory where you store CyberCortex.AI datastreams
path_to_destination_folder = r"D:/kitti2CyC/"

# Specify the dataset to load
date = '2011_09_26'
drives = ['0001', '0002', '0005', '0009', '0011', '0013', '0014', '0015', '0017', '0018', '0019', '0020', '0022',
          '0023', '0027', '0028', '0029', '0032', '0035', '0036', '0039', '0046', '0048', '0051', '0052', '0056',
          '0057', '0059', '0060', '0061', '0064', '0070', '0079', '0084', '0086', '0087', '0091', '0093']

category = {
    'city_': ['0091', '0093'],
    'residential_': ['0019', '0020', '0022', '0023', '0035', '0036', '0039', '0046', '0061', '0064', '0079', '0086',
                     '0087'],
    'road_': ['0015', '0027', '0028', '0029', '0032', '0052', '0070']
}

SENSORS = {'CAM_STEREO_COLOR': 0, 'CAM_STEREO_GRAY': 1, 'LIDAR': 2, 'IMU': 3, 'GPS': 4, 'STATE': 5, 'LABEL_2D': 6,
           "LABEL_3D": 7}
STEREO_SENSOR = [True, True, False, False, False, False, False, False]
SENSOR_ITERATOR = 0

# Flags
IS_2D_LABEL = False
IS_3D_LABEL = False

# --- Label Utils ---
TRACKLET_BOXES = []
TRACKLET_LABEL_INFO = []
# --- Bboxes vertex connections ---
VERTEX_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
    [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
    [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
]
AXES = [0, 1, 2]

ply_header_lidar = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property float r
end_header
'''

cyc_types_header = CFG.CyC_INFERENCE.TYPES_FILE


# create directory
def makedir(dir_path, dir_name):
    if os.path.isdir(dir_path + dir_name):
        print("Folder " + dir_path + dir_name + " already exist. Skipp command.")
    else:
        try:
            os.makedirs(dir_path + dir_name)
        except OSError:
            pass


# create folder structure for a given sensor type (e.g. Camera, Radar, Lidar)
def make_sensor(path_to_destination_folder, scene_name):
    global SENSOR_ITERATOR

    if not STEREO_SENSOR[SENSOR_ITERATOR]:
        makedir(
            path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                SENSOR_ITERATOR + 1) + '/', 'samples/')

        # move to the next sensor
        SENSOR_ITERATOR = SENSOR_ITERATOR + 1
    else:
        makedir(
            path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                SENSOR_ITERATOR + 1) + '/', 'samples/')
        makedir(
            path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                SENSOR_ITERATOR + 1) + '/samples/', 'left')
        makedir(
            path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                SENSOR_ITERATOR + 1) + '/samples/', 'right')

        # move to the next sensor
        SENSOR_ITERATOR = SENSOR_ITERATOR + 1

    return list()


def write_lidar_ply(fn, verts):
    """
    Write a PLY file. The data is stored as (x, y, z).
    :param verts: 3D point set.
    :return: None.
    """

    verts_num = len(verts)
    with open(fn, 'wb') as f:
        f.write((ply_header_lidar % dict(vert_num=verts_num)).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %f')


def generate_datablock_descriptor(list_sensors, drive):
    vision_core_id = 1

    cyc_types_path = CFG.CyC_INFERENCE.TYPES_FILE
    if not os.path.exists(cyc_types_path):
        print('CyC_TYPES.h: {0} not found'.format(cyc_types_path))
    CyC_FILTER_TYPE_ENUM_NAME = "CyC_FILTER_TYPE"
    CyC_DATA_TYPE_ENUM_NAME = "CyC_DATA_TYPE"

    filter_type = types_CyC_TYPES.get_datatypes_as_dict(cyc_types_path, CyC_FILTER_TYPE_ENUM_NAME)
    data_type = types_CyC_TYPES.get_datatypes_as_dict(cyc_types_path, CyC_DATA_TYPE_ENUM_NAME)

    SENSORS_FILTER = {
        'CAM_STEREO_COLOR': 'CyC_MONO_CAMERA_FILTER_TYPE',
        'CAM_STEREO_GRAY': 'CyC_MONO_CAMERA_FILTER_TYPE',
        'LIDAR': 'CyC_LIDAR_FILTER_TYPE',
        'IMU': 'CyC_IMU_FILTER_TYPE',
        'GPS': 'CyC_GPS_FILTER_TYPE',
        'STATE': 'CyC_VEHICLE_STATE_ESTIMATION_FILTER_TYPE',
        'LABEL_2D': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
        'LABEL_3D': 'CyC_OBJECT_DETECTOR_3D_FILTER_TYPE'
    }

    SENSORS_DATA = {
        'CAM_STEREO_COLOR': 'CyC_IMAGE',
        'CAM_STEREO_GRAY': 'CyC_IMAGE',
        'LIDAR': 'CyC_VOXELS',
        'IMU': 'CyC_IMU',
        'GPS': 'CyC_GPS',
        'STATE': 'CyC_STATE_MEASUREMENT',
        'LABEL_2D': 'CyC_2D_ROIS',
        'LABEL_3D': 'CyC_3D_BBOXES'
    }

    with open(path_to_destination_folder + '/' + date + '_drive_' + drive + '/datablock_descriptor.csv',
              mode='a',
              newline='') as blockchain_descriptor_file:
        blockchain_descriptor_writer = csv.writer(blockchain_descriptor_file, delimiter=',')
        header = list()
        header.append('vision_core_id')
        header.append('filter_id')
        header.append('name')
        header.append('type')
        header.append('output_data_type')
        header.append('input_sources')

        blockchain_descriptor_writer.writerow([column for column in header])

        for idx, sensor in enumerate(list_sensors):
            if 'CAM_STEREO_GRAY' in sensor:
                row = list()
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CAM_STEREO_GRAY')
                row.append(str(filter_type[SENSORS_FILTER['CAM_STEREO_GRAY']]))
                row.append(str(data_type[SENSORS_DATA['CAM_STEREO_GRAY']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'CAM_STEREO_COLOR' in sensor:
                row = list()
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CAM_STEREO_COLOR')
                row.append(str(filter_type[SENSORS_FILTER['CAM_STEREO_COLOR']]))
                row.append(str(data_type[SENSORS_DATA['CAM_STEREO_COLOR']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'LIDAR' in sensor:
                row = list()
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('LIDAR')
                row.append(str(filter_type[SENSORS_FILTER['LIDAR']]))
                row.append(str(data_type[SENSORS_DATA['LIDAR']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'IMU' in sensor:
                row = list()
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('IMU')
                row.append(str(filter_type[SENSORS_FILTER['IMU']]))
                row.append(str(data_type[SENSORS_DATA['IMU']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'STATE' in sensor:
                row = list()
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('STATE')
                row.append(str(filter_type[SENSORS_FILTER['STATE']]))
                row.append(str(data_type[SENSORS_DATA['STATE']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'LABEL_2D' in sensor:
                row = list()
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('LABEL_2D')
                row.append(str(filter_type[SENSORS_FILTER['LABEL_2D']]))
                row.append(str(data_type[SENSORS_DATA['LABEL_2D']]))
                row.append('{1-2' + ';1-' + str(idx) + '}')
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'LABEL_3D' in sensor:
                row = list()
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('LABEL_3D')
                row.append(str(filter_type[SENSORS_FILTER['LABEL_3D']]))
                row.append(str(data_type[SENSORS_DATA['LABEL_3D']]))
                row.append('{' + str(vision_core_id) + '-' + str(vision_core_id) + ';1-' + str(idx - 1) + '}')
                blockchain_descriptor_writer.writerow(column for column in row)


def get_corners(category, TRACKLET_LABEL_INFO, idx, sequence_index):
    def get_calib_cam_to_cam(cam_to_convert):
        '''
        function used to extract camera parameters
        :return: focal_distance_x, focal_distance_y, focal_center_x, focal_center_y
        '''

        cam_to_cam_calib_file_path = basedir + '/' + category + date + '/' + 'calib_cam_to_cam.txt'
        with open(cam_to_cam_calib_file_path, mode='r') as cam_to_cam_calib:
            size = list()
            R_rect_ = np.eye(4)
            R_rot_ = np.eye(4)
            for line in cam_to_cam_calib:
                if ('P_rect_' + cam_to_convert) in line:
                    image_transformation = line.split(" ")[1:]
                if ('R_rect_' + cam_to_convert) in line:
                    rotation_rect = line.split(" ")[1:]
                    for i in rotation_rect:
                        i = float(i)
                    R_rect_[:3, :3] = np.reshape(rotation_rect, newshape=(3, 3))
                if ('R_' + cam_to_convert) in line:
                    rotation = line.split(" ")[1:]
                    for i in rotation:
                        i = float(i)
                    R_rot_[:3, :3] = np.reshape(rotation, newshape=(3, 3))
                if ('S_rect_' + cam_to_convert) in line:
                    size.append(float(line.split(" ")[1]))
                    size.append(float(line.split(" ")[2]))
        return float(image_transformation[0]), float(image_transformation[5]), float(image_transformation[2]), float(
            image_transformation[6]), R_rect_, R_rot_, size

    fx_, fy_, cx_, cy_, R_rect, R_rot, size = get_calib_cam_to_cam('02')

    velo_pos = np.array([TRACKLET_LABEL_INFO[idx][sequence_index + 1][0],
                         TRACKLET_LABEL_INFO[idx][sequence_index + 1][1],
                         TRACKLET_LABEL_INFO[idx][sequence_index + 1][2],
                         1.0])

    w, h, l = TRACKLET_LABEL_INFO[idx][sequence_index + 2: sequence_index + 5]

    corners = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [0.0, 0.0, 0.0, 0.0, h, h, h, h],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ])

    rot_z = TRACKLET_LABEL_INFO[idx][sequence_index + 5][2]

    corners_rotation = np.array([
        [np.cos(rot_z), -np.sin(rot_z), 0.0, 0.0],
        [np.sin(rot_z), np.cos(rot_z), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    corners = np.dot(corners_rotation, corners)

    corners_translation = np.eye(4)
    corners_translation[:, 3] = velo_pos.T

    corners = np.dot(corners_translation, corners)

    translation_to_cam_02 = np.array([
        [1.0, 0.0, 0.0, -0.27],
        [0.0, 1.0, 0.0, -0.06],
        [0.0, 0.0, 1.0, 0.08],
        [0.0, 0.0, 0.0, 1.0]
    ])

    rotation_to_cam_02 = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    new_pos = np.dot(R_rect,
                     np.dot(R_rot, np.dot(rotation_to_cam_02, np.dot(translation_to_cam_02, corners))))

    x = fx_ * (new_pos[0] / new_pos[2]) + cx_
    y = fy_ * (new_pos[1] / new_pos[2]) + cy_

    x1, x2 = min(x), max(x)
    y1, y2 = min(y), max(y)

    x1 = x1 if x1 >= 0 else 0
    x1 = x1 if x1 < size[0] else size[0]

    x2 = x2 if x2 >= 0 else 0
    x2 = x2 if x2 < size[0] else size[0]

    y1 = y1 if y1 >= 0 else 0
    y1 = y1 if y1 < size[1] else size[1]

    y2 = y2 if y2 >= 0 else 0
    y2 = y2 if y2 < size[1] else size[1]

    return x1, x2, y1, y2


def map_data_2_CyC_format(dataset, category, drive, sensor):
    with open(path_to_destination_folder + '/' + date + '_drive_' + drive + '/datastream_1' + '_' +
              str(SENSORS[sensor] + 1) + '/data_descriptor.csv',
              mode='a',
              newline='') as data_descriptor_file:
        data_descriptor_writer = csv.writer(data_descriptor_file, delimiter=',')

        prev_timestamp = math.floor(datetime.timestamp(dataset.timestamps[0]) * 1000)  # convert to miliseconds
        sampling_time = 0
        # create headers
        header = []
        frame_based_sync_header = []
        header.append('timestamp_start')
        header.append('timestamp_stop')
        header.append('sampling_time')

        if 'CAM' in sensor:
            header.append('left_file_path_0')
            header.append('right_file_path_0')
        elif 'LIDAR' in sensor:
            header.append('lidar_file_path_0')
        elif 'IMU' in sensor:
            header.append('acc_x')
            header.append('acc_y')
            header.append('acc_z')
            header.append('gyro_x')
            header.append('gyro_y')
            header.append('gyro_z')
            header.append('pitch')
            header.append('yaw')
            header.append('roll')
        elif 'GPS' in sensor:
            header.append('lat')
            header.append('lon')
            header.append('alt')
        elif 'LABEL_2D' in sensor:
            header.append('frame_id')
            frame_based_sync_header.append('frame_id')
            frame_based_sync_header.append('roi_id')
            frame_based_sync_header.append('cls')
            frame_based_sync_header.append('x')
            frame_based_sync_header.append('y')
            frame_based_sync_header.append('width')
            frame_based_sync_header.append('height')
        elif 'LABEL_3D' in sensor:
            header.append('frame_id')
            frame_based_sync_header.append('frame_id')
            frame_based_sync_header.append('roi_id')
            frame_based_sync_header.append('cls')
            frame_based_sync_header.append('x')
            frame_based_sync_header.append('y')
            frame_based_sync_header.append('z')
            frame_based_sync_header.append('w')
            frame_based_sync_header.append('h')
            frame_based_sync_header.append('l')
            frame_based_sync_header.append('roll')
            frame_based_sync_header.append('pitch')
            frame_based_sync_header.append('yaw')
            frame_based_sync_header.append('alpha')
        else:
            header.append('state_variable_0')
            header.append('state_variable_1')
            header.append('state_variable_2')
            header.append('state_variable_3')
        data_descriptor_writer.writerow([column for column in header])
        total_label_rows = []
        for idx, timestamp in enumerate(dataset.timestamps):
            # compute the sampling_time
            if idx >= 1:
                sampling_time = math.floor(datetime.timestamp(dataset.timestamps[idx]) * 1000) - prev_timestamp  # [ms]

            if sensor is 'CAM_STEREO_GRAY':
                row = []
                # timestamp_start
                row.append(prev_timestamp)
                # timestamp_stop
                row.append(math.floor(datetime.timestamp(dataset.timestamps[idx]) * 1000))
                row.append(sampling_time)
                row.append('samples/left/' + os.path.basename(dataset.cam0_files[idx]))
                row.append('samples/right/' + os.path.basename(dataset.cam1_files[idx]))
                data_descriptor_writer.writerow(column for column in row)

            elif sensor is 'CAM_STEREO_COLOR':
                row = []
                # timestamp_start
                row.append(prev_timestamp)
                # timestamp_stop
                row.append(math.floor(datetime.timestamp(dataset.timestamps[idx]) * 1000))
                row.append(sampling_time)
                row.append('samples/left/' + os.path.basename(dataset.cam2_files[idx]))
                row.append('samples/right/' + os.path.basename(dataset.cam3_files[idx]))
                data_descriptor_writer.writerow(column for column in row)

            elif sensor is 'LIDAR':
                row = []
                # timestamp_start
                # timestamp_start
                row.append(prev_timestamp)
                # timestamp_stop
                row.append(math.floor(datetime.timestamp(dataset.timestamps[idx]) * 1000))
                row.append(sampling_time)
                row.append('samples/' + os.path.basename(dataset.velo_files[idx]) + '.ply')
                data_descriptor_writer.writerow(column for column in row)

            elif sensor in ['LABEL_2D', 'LABEL_3D']:
                row = []
                global IS_2D_LABEL
                global IS_3D_LABEL

                # TODO: make this oneline
                if sensor == 'LABEL_2D':
                    IS_2D_LABEL = True
                    IS_3D_LABEL = False
                else:
                    IS_3D_LABEL = True
                    IS_2D_LABEL = False


                # timestamp_start
                row.append(prev_timestamp)
                # timestamp_stop
                row.append(math.floor(datetime.timestamp(dataset.timestamps[idx]) * 1000))
                row.append(sampling_time)
                row.append(idx)
                sequence_index = 0
                for i in range(1, int(len(TRACKLET_LABEL_INFO[idx]) / 6) + 1):
                    x1, x2, y1, y2 = get_corners(category, TRACKLET_LABEL_INFO, idx, sequence_index)
                    if IS_2D_LABEL:
                        total_label_rows.append([
                            idx,
                            i,
                            kitti_classes_mapper[TRACKLET_LABEL_INFO[idx][sequence_index]][0],
                            int(x1),
                            int(y1),
                            int(x2 - x1),
                            int(y2 - y1)])
                    elif IS_3D_LABEL:
                        # Alpha observation angle calculation
                        theta_ray = np.arctan2(P2[0][0], (float(x2 + x1) / 2 - P2[0][2]))
                        yaw = TRACKLET_LABEL_INFO[idx][sequence_index + 5][2]
                        theta_l = (np.pi + yaw) + (np.pi + theta_ray)
                        new_alpha = float(-theta_l) + np.pi / 2.
                        if new_alpha < 0:
                            new_alpha += 2 * np.pi
                        new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)
                        new_alpha *= -1
                        total_label_rows.append([
                            idx,
                            i,
                            kitti_classes_mapper[TRACKLET_LABEL_INFO[idx][sequence_index]][0],
                            TRACKLET_LABEL_INFO[idx][sequence_index + 1][0],
                            TRACKLET_LABEL_INFO[idx][sequence_index + 1][1],
                            TRACKLET_LABEL_INFO[idx][sequence_index + 1][2],
                            TRACKLET_LABEL_INFO[idx][sequence_index + 2],
                            TRACKLET_LABEL_INFO[idx][sequence_index + 3],
                            TRACKLET_LABEL_INFO[idx][sequence_index + 4],
                            TRACKLET_LABEL_INFO[idx][sequence_index + 5][0],
                            TRACKLET_LABEL_INFO[idx][sequence_index + 5][1],
                            TRACKLET_LABEL_INFO[idx][sequence_index + 5][2],
                            new_alpha
                        ])
                    sequence_index = i * 6  # 6 values in a label list [label, translation, w, h, l, rotation]
                data_descriptor_writer.writerow(column for column in row)
            elif sensor is 'IMU':
                row = []
                # timestamp_start
                row.append(prev_timestamp)
                # timestamp_stop
                row.append(math.floor(datetime.timestamp(dataset.timestamps[idx]) * 1000))
                row.append(sampling_time)
                row.append(dataset.oxts[idx].packet.ax)
                row.append(dataset.oxts[idx].packet.ay)
                row.append(dataset.oxts[idx].packet.az)
                row.append(dataset.oxts[idx].packet.wx)
                row.append(dataset.oxts[idx].packet.wy)
                row.append(dataset.oxts[idx].packet.wz)
                row.append(dataset.oxts[idx].packet.pitch)
                row.append(dataset.oxts[idx].packet.yaw)
                row.append(dataset.oxts[idx].packet.roll)
                data_descriptor_writer.writerow(column for column in row)
            elif sensor is 'GPS':
                row = list()
                # timestamp_start
                row.append(prev_timestamp)
                # timestamp_stop
                row.append(math.floor(datetime.timestamp(dataset.timestamps[idx]) * 1000))
                row.append(sampling_time)
                row.append(dataset.oxts[idx].packet.lat)
                row.append(dataset.oxts[idx].packet.lon)
                row.append(dataset.oxts[idx].packet.alt)
                data_descriptor_writer.writerow(column for column in row)
            else:
                row = []
                # timestamp_star
                row.append(prev_timestamp)
                # timestamp_stop
                row.append(math.floor(datetime.timestamp(dataset.timestamps[idx]) * 1000))
                row.append(sampling_time)

                # get relative position (origin is the GPS coordinate of the first sample)
                t = dataset.oxts[idx].T_w_imu[:, -1]

                velocity_overall = 0

                # compute velocity of the vehicle
                if idx > 0:  # skip first measurement
                    pose_delta_x = t[0] - dataset.oxts[idx - 1].T_w_imu[:, -1][0]
                    pose_delta_y = t[1] - dataset.oxts[idx - 1].T_w_imu[:, -1][1]

                    time_delta = (math.floor(datetime.timestamp(dataset.timestamps[idx]) * 1000) - \
                                  math.floor(datetime.timestamp(dataset.timestamps[idx - 1]) * 1000)) / 1000  # [s]

                    if time_delta > 0:
                        velocity_x = pose_delta_x / time_delta
                        velocity_y = pose_delta_y / time_delta
                        velocity_overall = math.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)  # [m/s]

                row.append(t[0])  # x relative pose
                row.append(t[1])  # y relative pose
                row.append(velocity_overall)
                row.append(dataset.oxts[idx].packet.yaw)
                data_descriptor_writer.writerow(column for column in row)

            # current timestamp will be the previous timestamp in the next iteration
            prev_timestamp = math.floor(datetime.timestamp(dataset.timestamps[idx]) * 1000)
        if IS_2D_LABEL or IS_3D_LABEL:
            with open(path_to_destination_folder + '/' + date + '_drive_' + drive + '/datastream_1' + '_' +
                      str(SENSORS[sensor] + 1) + '/framebased_data_descriptor.csv',
                      mode='w+', newline='') as frame_based_descriptor:
                frame_based_descriptor_writer = csv.writer(frame_based_descriptor, delimiter=',')
                frame_based_descriptor_writer.writerow(col for col in frame_based_sync_header)
                for row in total_label_rows:
                    if IS_3D_LABEL:
                        frame_id, roi_id, cls, x, y, z, w, h, l, roll, pitch, yaw, alpha = row
                        frame_based_descriptor_writer.writerow(
                            [int(frame_id), int(roi_id), str(cls), x, y, z, w, h, l, roll, pitch, yaw, alpha])
                    elif IS_2D_LABEL:
                        frame_id, roi_id, cls, x, y, w, h = row
                        frame_based_descriptor_writer.writerow([int(frame_id), int(roi_id), str(cls), x, y, w, h])

    if sensor is 'CAM_STEREO_GRAY':
        # copy left images
        for src_file in dataset.cam0_files:
            copy(src_file, path_to_destination_folder + '/' + date + '_drive_' + drive + '/datastream_1' + '_' +
                 str(SENSORS['CAM_STEREO_GRAY'] + 1) + '/' + 'samples/left/')

        # copy right images
        for src_file in dataset.cam1_files:
            copy(src_file, path_to_destination_folder + '/' + date + '_drive_' + drive + '/datastream_1' + '_' +
                 str(SENSORS['CAM_STEREO_GRAY'] + 1) + '/' + 'samples/right/')

    elif sensor is 'CAM_STEREO_COLOR':
        # copy left images
        for src_file in dataset.cam2_files:
            copy(src_file, path_to_destination_folder + '/' + date + '_drive_' + drive + '/datastream_1' + '_' +
                 str(SENSORS['CAM_STEREO_COLOR'] + 1) + '/' + 'samples/left/')

        # copy right images
        for src_file in dataset.cam3_files:
            copy(src_file, path_to_destination_folder + '/' + date + '_drive_' + drive + '/datastream_1' + '_' +
                 str(SENSORS['CAM_STEREO_COLOR'] + 1) + '/' + 'samples/right/')

    elif sensor is 'LIDAR':
        # save Velodyne data in PLY format (such that CyberCortex.AI can use standard libraries to loade it)
        for idx, velo_file in enumerate(dataset.velo_files):
            write_lidar_ply(
                path_to_destination_folder + '/' + date + '_drive_' + drive + '/datastream_1' + '_' +
                str(SENSORS['LIDAR'] + 1) + '/samples/' + os.path.basename(velo_file) + '.ply',
                dataset.get_velo(idx))


def search_timestamp(timestamp, dataset):
    # search a timestamp in the first column of the list
    for row in dataset.timestamps:
        # if timestamps match
        if timestamp == math.floor(datetime.timestamp(row) * 1000):
            return True


def generate_sync_descriptor(dataset, drive):
    with open(path_to_destination_folder + '/' + date + '_drive_' + drive + '/sampling_timestamps_sync.csv',
              mode='a',
              newline='') as data_descriptor_file:
        data_descriptor_writer = csv.writer(data_descriptor_file, delimiter=',')

        # write header
        header = []
        header.append('timestamp_stop')
        for it in range(len(SENSORS)):
            header.append('datastream_1' + '_' + str(it + 1))
        data_descriptor_writer.writerow([column for column in header])

        # fill out sync file with timestamps every 1 ms and -1 in all datastreams
        for timestamp in range(math.floor(datetime.timestamp(dataset.timestamps[0]) * 1000),
                               math.floor(datetime.timestamp(dataset.timestamps[-1]) * 1000) + 1):
            row = []
            row.append(timestamp)

            if search_timestamp(timestamp, dataset) is True:
                for it in range(len(SENSORS)):
                    row.append(timestamp)
            else:
                for it in range(len(SENSORS)):
                    row.append(-1)

            data_descriptor_writer.writerow(column for column in row)


def main():
    for key in category.keys():
        for drive in category[key]:
            global SENSOR_ITERATOR
            SENSOR_ITERATOR = 0

            # create folder of the current scene
            makedir(path_to_destination_folder, date + '_drive_' + drive)

            # create folder structure for each sensor
            # add each sensor to a list
            for sensor in range(len(SENSORS)):
                make_sensor(path_to_destination_folder, date + '_drive_' + drive)

            # Load the data. Optionally, specify the frame range to load.
            dataset = pykitti.raw(basedir, (key + date), drive)
            # Load data for labels (tracklets provided with the dataset)
            global TRACKLET_LABEL_INFO
            _boxes, TRACKLET_LABEL_INFO = load_tracklets_for_frames(len(list(dataset.velo)),
                                                                    basedir + '/' + key + date + '/' + key + date + '_drive_' + drive + '_sync' + '/' + 'tracklet_labels.xml')
            generate_datablock_descriptor(SENSORS, drive)

            for idx, sensor, in enumerate(SENSORS):
                map_data_2_CyC_format(dataset, key, drive, sensor)

            generate_sync_descriptor(dataset, drive)


if __name__ == "__main__":
    main()
