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
Install:
- pytorch as described here: https://pytorch.org/
- NuScenes devkit: pip install nuscenes-devkit
- cachetools
"""

import csv
import json
import math
import os
import numpy as np
from shutil import copy
from box_utils import get_boxes, find_sample, get_corners
from cloud_utils import load_lidar_from_file, write_lidar_ply, load_radar_from_file, write_radar_ply
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
import varname
from data import types_CyC_TYPES
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

# path to json folder
path_to_database_folder = r"C:/dev/NuScenes/"
path_to_annotation_folder = path_to_database_folder + "/v1.0-mini"
path_to_destination_folder = r"C:/NuScenes"

scene_file_path = path_to_annotation_folder + "/scene.json"
sample_file_path = path_to_annotation_folder + "/sample.json"
sample_data_file_path = path_to_annotation_folder + "/sample_data.json"
ego_pose_file_path = path_to_annotation_folder + "/ego_pose.json"
instance_file_path = path_to_annotation_folder + "/instance.json"
category_file_path = path_to_annotation_folder + "/category.json"
annotations_file_path = path_to_annotation_folder + "/sample_annotation.json"
calibration_file_path = path_to_annotation_folder + "/calibrated_sensor.json"

bCopy_only_key_frame = False  # save sensory data every 500 [ms]

SENSORS = {'CAM_FRONT': 0, 'CAM_FRONT_LEFT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK': 3,
           'CAM_BACK_LEFT': 4, 'CAM_BACK_RIGHT': 5, 'LIDAR_TOP': 6, 'RADAR_FRONT': 7,
           'RADAR_FRONT_LEFT': 8, 'RADAR_FRONT_RIGHT': 9, 'RADAR_BACK_LEFT': 10, 'RADAR_BACK_RIGHT': 11,
           'State': 12, 'IMU': 13, '2D_CAM_FRONT': 14, '2D_CAM_FRONT_LEFT': 15, '2D_CAM_FRONT_RIGHT': 16,
           '2D_CAM_BACK': 17, '2D_CAM_BACK_LEFT': 18, '2D_CAM_BACK_RIGHT': 19, '3D_CAM_FRONT': 20,
           '3D_CAM_FRONT_LEFT': 21, '3D_CAM_FRONT_RIGHT': 22, '3D_CAM_BACK': 23, '3D_CAM_BACK_LEFT': 24,
           '3D_CAM_BACK_RIGHT': 25}


CAMERA_SENSOR = [
    True, True, True, True,
    True, True, False, False,
    False, False, False, False,
    False, False, False, False,
    False, False, False, False,
    False, False, False, False,
    False, False
]


SENSOR_ITERATOR = 0

CyC_types_header = os.path.join(os.path.dirname(__file__), "filters\CyC_TYPES.h")


# convert classes to IDs
def class_string_to_id(class_string):
    class_dict = {
        'human.pedestrian.adult': 1,
        'human.pedestrian.child': 2,
        'human.pedestrian.construction_worker': 3,
        'human.pedestrian.personal_mobility': 4,
        'human.pedestrian.police_officer': 5,
        'movable_object.barrier': 6,
        'movable_object.debris': 7,
        'movable_object.pushable_pullable': 8,
        'movable_object.trafficcone': 9,
        'static_object.bicycle_rack': 10,
        'vehicle.bicycle': 11,
        'vehicle.bus.bendy': 12,
        'vehicle.bus.rigid': 13,
        'vehicle.car': 14,
        'vehicle.construction': 15,
        'vehicle.motorcycle': 16,
        'vehicle.trailer': 17,
        'vehicle.truck': 18,
    }

    return class_dict[class_string]


# create directory
def makedir(dir_path, dir_name):
    if os.path.isdir(dir_path + dir_name):
        print("Folder " + dir_path + dir_name + " already exist. Skipp command.")
    else:
        try:
            os.makedirs(dir_path + dir_name)
        except OSError:
            pass


def get_sample_data(json_object, sample_token):
    sample_data = []
    for sensor_sample in json_object:
        if sensor_sample['sample_token'] == sample_token:
            sample_data.append(sensor_sample)

    return sample_data


# create folder structure for a given sensor type (e.g. Camera, Radar, Lidar)
def make_sensor(path_to_destination_folder, scene_name):
    global SENSOR_ITERATOR

    if not CAMERA_SENSOR[SENSOR_ITERATOR]:
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


def get_state(prev_sensor, sensor, prev_sensor_ego_pose, sensor_ego_pose):
    yaw = Quaternion(sensor_ego_pose['rotation']).yaw_pitch_roll[0]
    time_delta = (1e-6 * sensor['timestamp']) - (
            1e-6 * prev_sensor['timestamp'])  # 1e-6 is for converting from uSec to sec
    pose_delta_x = sensor_ego_pose['translation'][0] - prev_sensor_ego_pose['translation'][0]
    pose_delta_y = sensor_ego_pose['translation'][1] - prev_sensor_ego_pose['translation'][1]

    velocity_overall = 0
    if time_delta > 0:
        velocity_x = pose_delta_x / time_delta
        velocity_y = pose_delta_y / time_delta
        velocity_overall = math.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)

    return {'timestamp': sensor['timestamp'], 'x': sensor_ego_pose['translation'][0],
            'y': sensor_ego_pose['translation'][1], 'yaw': yaw, 'velocity': velocity_overall}


def process_scene(list_sensors, IMU_data):
    if len(IMU_data) > 0:
        for imu_data in IMU_data:
            old_keyname = 'utime'
            new_keyname = 'timestamp'
            imu_data[new_keyname] = imu_data.pop(old_keyname)
            list_sensors[SENSORS['IMU']].append(imu_data)


# copy sample file and add the information to a list for further processing
def process_sample(path_to_destination_folder, scene_name, prev_sensor, sensor, prev_sensor_ego_pose, sensor_ego_pose, list_sensors, sample_annotations, frame_idx):
    # get the sensor type
    path, filename = os.path.split(sensor['filename'])
    sensor_type = os.path.split(os.path.dirname(sensor['filename']))[-1]

    # append sample data to the list
    list_sensors[SENSORS[sensor_type]].append({'timestamp': sensor['timestamp'],
                                               'filename': 'samples/' + 'left/' + filename if CAMERA_SENSOR[
                                                   SENSORS[sensor_type]] else 'samples/' + filename})
    if len(sample_annotations) > 0:
        channel = os.path.split(os.path.dirname(sample_annotations[0]))[-1]
        if 'CAM' in channel:
            if '2D_'+channel in SENSORS:
                list_sensors[SENSORS['2D_' + channel]].append({
                    'timestamp': sensor['timestamp'],
                    'channel': '2D_' + channel,
                    'frame_index': frame_idx,
                    'boxes': [box for box in sample_annotations[1]]})
            if '3D_'+channel in SENSORS:
                list_sensors[SENSORS['3D_' + channel]].append(
                    {'timestamp': sensor['timestamp'],
                     'channel': '3D_' + channel,
                     'frame_index': frame_idx,
                     'boxes': [box for box in sample_annotations[1]]})

        # append the state only for the camera front sensor
        if sensor_type == 'CAM_FRONT':
            list_sensors[SENSORS['State']].append(get_state(prev_sensor, sensor, prev_sensor_ego_pose, sensor_ego_pose))

            src = path_to_database_folder + sensor['filename']
            dst = path_to_destination_folder + '/' + scene_name + '/datastream_1_' + str(SENSORS[sensor_type] + 1) + \
                  '/samples/left/'
            copy(src, dst)
        elif sensor_type == 'CAM_FRONT_LEFT':
            list_sensors[SENSORS['State']].append(get_state(prev_sensor, sensor, prev_sensor_ego_pose, sensor_ego_pose))

            src = path_to_database_folder + sensor['filename']
            dst = path_to_destination_folder + '/' + scene_name + '/datastream_1_' + str(SENSORS[sensor_type] + 1) + \
                  '/samples/left/'
            copy(src, dst)
        elif sensor_type == 'CAM_FRONT_RIGHT':
            list_sensors[SENSORS['State']].append(get_state(prev_sensor, sensor, prev_sensor_ego_pose, sensor_ego_pose))

            src = path_to_database_folder + sensor['filename']
            dst = path_to_destination_folder + '/' + scene_name + '/datastream_1_' + str(SENSORS[sensor_type] + 1) + \
                  '/samples/left/'
            copy(src, dst)
        elif sensor_type == 'CAM_BACK':
            list_sensors[SENSORS['State']].append(get_state(prev_sensor, sensor, prev_sensor_ego_pose, sensor_ego_pose))

            src = path_to_database_folder + sensor['filename']
            dst = path_to_destination_folder + '/' + scene_name + '/datastream_1_' + str(SENSORS[sensor_type] + 1) + \
                  '/samples/left/'
            copy(src, dst)
        elif sensor_type == 'CAM_BACK_LEFT':
            list_sensors[SENSORS['State']].append(get_state(prev_sensor, sensor, prev_sensor_ego_pose, sensor_ego_pose))

            src = path_to_database_folder + sensor['filename']
            dst = path_to_destination_folder + '/' + scene_name + '/datastream_1_' + str(SENSORS[sensor_type] + 1) + \
                  '/samples/left/'
            copy(src, dst)
        elif sensor_type == 'CAM_BACK_RIGHT':
            list_sensors[SENSORS['State']].append(get_state(prev_sensor, sensor, prev_sensor_ego_pose, sensor_ego_pose))

            src = path_to_database_folder + sensor['filename']
            dst = path_to_destination_folder + '/' + scene_name + '/datastream_1_' + str(SENSORS[sensor_type] + 1) + \
                  '/samples/left/'
            copy(src, dst)
        elif 'LIDAR' in sensor_type:
            # load lidar bin file
            lidar_coud = load_lidar_from_file(path_to_database_folder + sensor['filename'])

            # save directly into the destination folder
            write_lidar_ply(
                  path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                    SENSORS[sensor_type] + 1) + '/' + 'samples/' + filename + '.ply', lidar_coud)
        elif 'RADAR' in sensor_type:
            radar_cloud = load_radar_from_file(path_to_database_folder + sensor['filename'])

            # save directly into the destination folder
            write_radar_ply(
                path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                    SENSORS[sensor_type] + 1) + '/' + 'samples/' + filename + '.ply', radar_cloud)


def generate_blockchain_descriptor(list_sensors, scene_name):
    vision_core_id = 1

    CyC_types_path = r"C:\dev\src\CyberCortex.AI\core\include\CyC_TYPES.h"
    if not os.path.exists(CyC_types_path):
        print('CyC_TYPES.h: {0} not found'.format(CyC_types_path))
    CyC_FILTER_TYPE_ENUM_NAME = "CyC_FILTER_TYPE"
    CyC_DATA_TYPE_ENUM_NAME = "CyC_DATA_TYPE"

    filter_type = types_CyC_TYPES.get_datatypes_as_dict(CyC_types_path, CyC_FILTER_TYPE_ENUM_NAME)
    data_type = types_CyC_TYPES.get_datatypes_as_dict(CyC_types_path, CyC_DATA_TYPE_ENUM_NAME)

    SENSORS_FILTER = {'CAM_FRONT': 'CyC_MONO_CAMERA_FILTER_TYPE',
                      'CAM_FRONT_LEFT': 'CyC_MONO_CAMERA_FILTER_TYPE',
                      'CAM_FRONT_RIGHT': 'CyC_MONO_CAMERA_FILTER_TYPE',
                      'CAM_BACK': 'CyC_MONO_CAMERA_FILTER_TYPE',
                      'CAM_BACK_LEFT': 'CyC_MONO_CAMERA_FILTER_TYPE',
                      'CAM_BACK_RIGHT': 'CyC_MONO_CAMERA_FILTER_TYPE',
                      'LIDAR_TOP': 'CyC_LIDAR_FILTER_TYPE',
                      'RADAR_FRONT': 'CyC_RADAR_FILTER_TYPE',
                      'RADAR_FRONT_LEFT': 'CyC_RADAR_FILTER_TYPE',
                      'RADAR_FRONT_RIGHT': 'CyC_RADAR_FILTER_TYPE',
                      'RADAR_BACK_LEFT': 'CyC_RADAR_FILTER_TYPE',
                      'RADAR_BACK_RIGHT': 'CyC_RADAR_FILTER_TYPE',
                      'VehicleOdometry': 'CyC_VEHICLE_ODOMETRY_FILTER_TYPE',
                      '2D_CAM_FRONT': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '2D_CAM_FRONT_LEFT': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '2D_CAM_FRONT_RIGHT': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '2D_CAM_BACK': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '2D_CAM_BACK_LEFT': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '2D_CAM_BACK_RIGHT': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '3D_CAM_FRONT': 'CyC_OBJECT_DETECTOR_3D_FILTER_TYPE',
                      '3D_CAM_FRONT_LEFT': 'CyC_OBJECT_DETECTOR_3D_FILTER_TYPE',
                      '3D_CAM_FRONT_RIGHT': 'CyC_OBJECT_DETECTOR_3D_FILTER_TYPE',
                      '3D_CAM_BACK': 'CyC_OBJECT_DETECTOR_3D_FILTER_TYPE',
                      '3D_CAM_BACK_LEFT': 'CyC_OBJECT_DETECTOR_3D_FILTER_TYPE',
                      '3D_CAM_BACK_RIGHT': 'CyC_OBJECT_DETECTOR_3D_FILTER_TYPE',
                      'IMU': 'CyC_IMU_FILTER_TYPE'
                      }

    SENSORS_DATA = {'CAM_FRONT': 'CyC_IMAGE',
                    'CAM_FRONT_LEFT': 'CyC_IMAGE',
                    'CAM_FRONT_RIGHT': 'CyC_IMAGE',
                    'CAM_BACK': 'CyC_IMAGE',
                    'CAM_BACK_LEFT': 'CyC_IMAGE',
                    'CAM_BACK_RIGHT': 'CyC_IMAGE',
                    'LIDAR_TOP': 'CyC_VOXELS',
                    'RADAR_FRONT': 'CyC_RADAR',
                    'RADAR_FRONT_LEFT': 'CyC_RADAR',
                    'RADAR_FRONT_RIGHT': 'CyC_RADAR',
                    'RADAR_BACK_LEFT': 'CyC_RADAR',
                    'RADAR_BACK_RIGHT': 'CyC_RADAR',
                    'VehicleOdometry': 'CyC_STATE_MEASUREMENT',
                    '2D_CAM_FRONT': 'CyC_2D_ROIS',
                    '2D_CAM_FRONT_LEFT': 'CyC_2D_ROIS',
                    '2D_CAM_FRONT_RIGHT': 'CyC_2D_ROIS',
                    '2D_CAM_BACK': 'CyC_2D_ROIS',
                    '2D_CAM_BACK_LEFT': 'CyC_2D_ROIS',
                    '2D_CAM_BACK_RIGHT': 'CyC_2D_ROIS',
                    '3D_CAM_FRONT': 'CyC_3D_BBOXES',
                    '3D_CAM_FRONT_LEFT': 'CyC_3D_BBOXES',
                    '3D_CAM_FRONT_RIGHT': 'CyC_3D_BBOXES',
                    '3D_CAM_BACK': 'CyC_3D_BBOXES',
                    '3D_CAM_BACK_LEFT': 'CyC_3D_BBOXES',
                    '3D_CAM_BACK_RIGHT': 'CyC_3D_BBOXES',
                    'IMU': 'CyC_IMU'
                    }

    with open(path_to_destination_folder + '/' + scene_name + '/datablock_descriptor.csv', mode='a',
              newline='') as blockchain_descriptor_file:
        blockchain_descriptor_writer = csv.writer(blockchain_descriptor_file, delimiter=',')

        header = []
        header.append('vision_core_id')
        header.append('filter_id')
        header.append('name')
        header.append('type')
        header.append('output_data_type')
        header.append('input_sources')

        blockchain_descriptor_writer.writerow([column for column in header])

        objdet_row = {
            '2D_CAM_FRONT': [],
            '2D_CAM_FRONT_LEFT': [],
            '2D_CAM_FRONT_RIGHT': [],
            '2D_CAM_BACK': [],
            '2D_CAM_BACK_LEFT': [],
            '2D_CAM_BACK_RIGHT': [],
            '3D_CAM_FRONT': [],
            '3D_CAM_FRONT_LEFT': [],
            '3D_CAM_FRONT_RIGHT': [],
            '3D_CAM_BACK': [],
            '3D_CAM_BACK_LEFT': [],
            '3D_CAM_BACK_RIGHT': []
        }

        for idx, sensor in enumerate(list_sensors):
            if list(sensor[0].keys())[1] == 'filename':
                sensor_type = os.path.basename(sensor[0]['filename'])
            elif list(sensor[0].keys())[1] == 'channel':
                if '2D' in sensor[0]['channel']:
                    sensor_type = '2D'
                elif '3D' in sensor[0]['channel']:
                    sensor_type = '3D'
            elif list(sensor[0].keys())[0] == 'linear_accel':
                sensor_type = 'IMU'
            else:
                sensor_type = 'VehicleOdometry'

            if 'VehicleOdometry' in sensor_type:
                row = list()
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('VehicleOdometry')
                row.append(str(filter_type[SENSORS_FILTER['VehicleOdometry']]))
                row.append(str(data_type[SENSORS_DATA['VehicleOdometry']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'IMU' in sensor_type:
                row = list()
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('IMU')
                row.append(str(filter_type[SENSORS_FILTER['IMU']]))
                row.append(str(data_type[SENSORS_DATA['IMU']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'CAM_FRONT_LEFT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CameraFrontLeft')
                print(filter_type[SENSORS_FILTER['CAM_FRONT_LEFT']])
                row.append(str(filter_type[SENSORS_FILTER['CAM_FRONT_LEFT']]))
                row.append(str(data_type[SENSORS_DATA['CAM_FRONT_LEFT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                objdet_row[('2D_' + 'CAM_FRONT_LEFT')].append(str(vision_core_id))
                objdet_row[('2D_' + 'CAM_FRONT_LEFT')].append(str(idx + 15))
                objdet_row[('2D_' + 'CAM_FRONT_LEFT')].append('ObjectDetector2D_'+str('CAM_FRONT_LEFT'))
                objdet_row[('2D_' + 'CAM_FRONT_LEFT')].append(str(filter_type[SENSORS_FILTER['2D_CAM_FRONT_LEFT']]))
                objdet_row[('2D_' + 'CAM_FRONT_LEFT')].append(str(data_type[SENSORS_DATA['2D_CAM_FRONT_LEFT']]))
                objdet_row[('2D_' + 'CAM_FRONT_LEFT')].append("{1-1;1-12}")
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append(str(idx + 21))
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append('ObjectDetector3D_' + str('CAM_FRONT_LEFT'))
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append(str(filter_type[SENSORS_FILTER['3D_CAM_FRONT_LEFT']]))
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append(str(data_type[SENSORS_DATA['3D_CAM_FRONT_LEFT']]))
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append("{1-1;1-12}")
            elif 'CAM_FRONT_RIGHT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CameraFrontRight')
                row.append(str(filter_type[SENSORS_FILTER['CAM_FRONT_RIGHT']]))
                row.append(str(data_type[SENSORS_DATA['CAM_FRONT_RIGHT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append(str(vision_core_id))
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append(str(idx + 15))
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append('ObjectDetector2D_' + str('CAM_FRONT_RIGHT'))
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append(str(filter_type[SENSORS_FILTER['2D_CAM_FRONT_RIGHT']]))
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append(str(data_type[SENSORS_DATA['2D_CAM_FRONT_RIGHT']]))
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append("{1-1;1-12}")
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append(str(idx + 21))
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append('ObjectDetector3D_' + str('CAM_FRONT_RIGHT'))
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append(str(filter_type[SENSORS_FILTER['3D_CAM_FRONT_RIGHT']]))
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append(str(data_type[SENSORS_DATA['3D_CAM_FRONT_RIGHT']]))
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append("{1-1;1-12}")
            elif 'CAM_FRONT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CameraFront')
                row.append(str(filter_type[SENSORS_FILTER['CAM_FRONT']]))
                row.append(str(data_type[SENSORS_DATA['CAM_FRONT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                objdet_row[('2D_' + 'CAM_FRONT')].append(str(vision_core_id))
                objdet_row[('2D_' + 'CAM_FRONT')].append(str(idx + 15))
                objdet_row[('2D_' + 'CAM_FRONT')].append('ObjectDetector2D_' + str('CAM_FRONT'))
                objdet_row[('2D_' + 'CAM_FRONT')].append(str(filter_type[SENSORS_FILTER['2D_CAM_FRONT']]))
                objdet_row[('2D_' + 'CAM_FRONT')].append(str(data_type[SENSORS_DATA['2D_CAM_FRONT']]))
                objdet_row[('2D_' + 'CAM_FRONT')].append("{1-1;1-12}")
                objdet_row[('3D_' + 'CAM_FRONT')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_FRONT')].append(str(idx + 21))
                objdet_row[('3D_' + 'CAM_FRONT')].append('ObjectDetector3D_' + str('CAM_FRONT'))
                objdet_row[('3D_' + 'CAM_FRONT')].append(str(filter_type[SENSORS_FILTER['3D_CAM_FRONT']]))
                objdet_row[('3D_' + 'CAM_FRONT')].append(str(data_type[SENSORS_DATA['3D_CAM_FRONT']]))
                objdet_row[('3D_' + 'CAM_FRONT')].append("{1-1;1-12}")
            elif 'CAM_BACK_LEFT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CameraBackLeft')
                row.append(str(filter_type[SENSORS_FILTER['CAM_BACK_LEFT']]))
                row.append(str(data_type[SENSORS_DATA['CAM_BACK_LEFT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append(str(vision_core_id))
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append(str(idx + 15))
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append('ObjectDetector2D_' + str('CAM_BACK_LEFT'))
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append(str(filter_type[SENSORS_FILTER['2D_CAM_BACK_LEFT']]))
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append(str(data_type[SENSORS_DATA['2D_CAM_BACK_LEFT']]))
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append("{1-1;1-12}")
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append(str(idx + 21))
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append('ObjectDetector3D_' + str('CAM_BACK_LEFT'))
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append(str(filter_type[SENSORS_FILTER['3D_CAM_BACK_LEFT']]))
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append(str(data_type[SENSORS_DATA['3D_CAM_BACK_LEFT']]))
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append("{1-1;1-12}")
            elif 'CAM_BACK_RIGHT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CameraBackRight')
                row.append(str(filter_type[SENSORS_FILTER['CAM_BACK_RIGHT']]))
                row.append(str(data_type[SENSORS_DATA['CAM_BACK_RIGHT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append(str(vision_core_id))
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append(str(idx + 15))
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append('ObjectDetector2D_' + str('CAM_BACK_RIGHT'))
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append(str(filter_type[SENSORS_FILTER['2D_CAM_BACK_RIGHT']]))
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append(str(data_type[SENSORS_DATA['2D_CAM_BACK_RIGHT']]))
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append("{1-1;1-12}")
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append(str(idx + 21))
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append('ObjectDetector3D_' + str('CAM_BACK_RIGHT'))
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append(str(filter_type[SENSORS_FILTER['3D_CAM_BACK_RIGHT']]))
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append(str(data_type[SENSORS_DATA['3D_CAM_BACK_RIGHT']]))
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append("{1-1;1-12}")
            elif 'CAM_BACK' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CameraBack')
                row.append(str(filter_type[SENSORS_FILTER['CAM_BACK']]))
                row.append(str(data_type[SENSORS_DATA['CAM_BACK']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                objdet_row[('2D_' + 'CAM_BACK')].append(str(vision_core_id))
                objdet_row[('2D_' + 'CAM_BACK')].append(str(idx + 15))
                objdet_row[('2D_' + 'CAM_BACK')].append('ObjectDetector2D_' + str('CAM_BACK'))
                objdet_row[('2D_' + 'CAM_BACK')].append(str(filter_type[SENSORS_FILTER['2D_CAM_BACK']]))
                objdet_row[('2D_' + 'CAM_BACK')].append(str(data_type[SENSORS_DATA['2D_CAM_BACK']]))
                objdet_row[('2D_' + 'CAM_BACK')].append("{1-1;1-12}")
                objdet_row[('3D_' + 'CAM_BACK')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_BACK')].append(str(idx + 21))
                objdet_row[('3D_' + 'CAM_BACK')].append('ObjectDetector3D_' + str('CAM_BACK'))
                objdet_row[('3D_' + 'CAM_BACK')].append(str(filter_type[SENSORS_FILTER['3D_CAM_BACK']]))
                objdet_row[('3D_' + 'CAM_BACK')].append(str(data_type[SENSORS_DATA['3D_CAM_BACK']]))
                objdet_row[('3D_' + 'CAM_BACK')].append("{1-1;1-12}")
            elif 'LIDAR_TOP' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('LidarTop')
                row.append(str(filter_type[SENSORS_FILTER['LIDAR_TOP']]))
                row.append(str(data_type[SENSORS_DATA['LIDAR_TOP']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'RADAR_FRONT_LEFT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('RadarFrontLeft')
                row.append(str(filter_type[SENSORS_FILTER['RADAR_FRONT_LEFT']]))
                row.append(str(data_type[SENSORS_DATA['RADAR_FRONT_LEFT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'RADAR_FRONT_RIGHT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('RadarFrontRight')
                row.append(str(filter_type[SENSORS_FILTER['RADAR_FRONT_RIGHT']]))
                row.append(str(data_type[SENSORS_DATA['RADAR_FRONT_RIGHT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'RADAR_FRONT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('RadarFront')
                row.append(str(filter_type[SENSORS_FILTER['RADAR_FRONT']]))
                row.append(str(data_type[SENSORS_DATA['RADAR_FRONT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'RADAR_BACK_LEFT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('RadarBackLeft')
                row.append(str(filter_type[SENSORS_FILTER['RADAR_BACK_LEFT']]))
                row.append(str(data_type[SENSORS_DATA['RADAR_BACK_LEFT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
            elif 'RADAR_BACK_RIGHT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('RadarBackRight')
                row.append(str(filter_type[SENSORS_FILTER['RADAR_BACK_RIGHT']]))
                row.append(str(data_type[SENSORS_DATA['RADAR_BACK_RIGHT']]))
                blockchain_descriptor_writer.writerow(column for column in row)

        for key in objdet_row.keys():
            if len(objdet_row[key]):
                blockchain_descriptor_writer.writerow(column for column in objdet_row[key])


def get_sample_filename_by_timestamp(json_object, timestamp):
    for sample in json_object:
        if sample['timestamp'] == timestamp and 'CAM' in sample['filename']:
            return sample['filename']


def generate_data_descriptor(list_sensors, scene_name, annotations_data, sample_data, calibration_data):
    for idx, sensor in enumerate(list_sensors):
        if len(sensor) > 0:
            with open(path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(idx + 1) +
                      '/data_descriptor.csv',
                      mode='a',
                      newline='') as data_descriptor_file:
                data_descriptor_writer = csv.writer(data_descriptor_file, delimiter=',')
                total_label_rows = []
                prev_timestamp = sensor[0]['timestamp']
                sampling_time = 0
                # create headers
                header = []
                header.append('timestamp_start')
                header.append('timestamp_stop')
                header.append('sampling_time')

                frame_based_sync_header = []

                # get the sensor type
                if list(sensor[0].keys())[1] == 'filename':
                    sensor_type = os.path.basename(sensor[0]['filename'])
                elif list(sensor[0].keys())[1] == 'channel':
                    sensor_type = sensor[0]['channel']
                elif list(sensor[0].keys())[0] == 'linear_accel':
                    sensor_type = 'IMU'
                else:
                    sensor_type = 'State'

                print('Processing datastream_1_' + str(idx + 1))

                if '2D' in sensor_type:
                    # print('Processing datastream_1_' + str(SENSORS[sensor_type]))
                    header.append('frame_id')
                    frame_based_sync_header.append('frame_id')
                    frame_based_sync_header.append('roi_id')
                    frame_based_sync_header.append('cls')
                    frame_based_sync_header.append('x')
                    frame_based_sync_header.append('y')
                    frame_based_sync_header.append('width')
                    frame_based_sync_header.append('height')
                # create header depending on the type of sensor
                elif '3D' in sensor_type:
                    # print('Processing' + sensor_type)
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
                elif 'CAM' in sensor_type:
                    header.append('left_file_path_0')
                    header.append('right_file_path_0')
                elif 'LIDAR' in sensor_type:
                    header.append('lidar_file_path_0')
                elif 'RADAR' in sensor_type:
                    header.append('radar_file_path_0')
                elif 'IMU' in sensor_type:
                    header.append('acc_x')
                    header.append('acc_y')
                    header.append('acc_z')
                    header.append('gyro_x')
                    header.append('gyro_y')
                    header.append('gyro_z')
                    header.append('pitch')
                    header.append('yaw')
                    header.append('roll')
                else:
                    header.append('state_variable_0')
                    header.append('state_variable_1')
                    header.append('state_variable_2')
                    header.append('state_variable_3')
                data_descriptor_writer.writerow([column for column in header])

                # create a row for each timestamp of the first sensor type
                for idx_sample, sample in enumerate(sensor):
                    # compute the sampling_time
                    if idx_sample >= 1:
                        sampling_time = int(int(abs(sample['timestamp'] - prev_timestamp)) / 1000)  # [ms]

                    if '2D' in sensor_type:
                        row = []
                        roi_index = 0
                        # timestamp_start
                        row.append(math.floor(prev_timestamp / 1000))
                        # timestamp_stop
                        row.append(math.floor(sample['timestamp'] / 1000))
                        row.append(sampling_time)
                        row.append(sample['frame_index'])
                        data_descriptor_writer.writerow(column for column in row)
                        for box in sample['boxes']:
                            corners = get_corners(box, sample, json_annotations=annotations_data, json_sample_data=sample_data, json_calibration=calibration_data)

                            x1 = int(min(corners.T[:, 0]))
                            y1 = int(min(corners.T[:, 1]))

                            x2 = int(max(corners.T[:, 0]))
                            y2 = int(max(corners.T[:, 1]))

                            x1 = x1 if x1 >= 0 else 0
                            x1 = x1 if x1 < 1600 else 1600

                            x2 = x2 if x2 >= 0 else 0
                            x2 = x2 if x2 < 1600 else 1600

                            y1 = y1 if y1 >= 0 else 0
                            y1 = y1 if y1 < 900 else 900

                            y2 = y2 if y2 >= 0 else 0
                            y2 = y2 if y2 < 900 else 900

                            total_label_rows.append([sample['frame_index'], roi_index, class_string_to_id(box.name), x1, y1 , (x2 - x1), (y2 - y1)])
                            roi_index += 1

                    elif '3D' in sensor_type:
                        row = []
                        roi_index = 0
                        # timestamp_start
                        row.append(math.floor(prev_timestamp / 1000))
                        # timestamp_stop
                        row.append(math.floor(sample['timestamp'] / 1000))
                        row.append(sampling_time)
                        row.append(sample['frame_index'])
                        data_descriptor_writer.writerow(column for column in row)
                        for box in sample['boxes']:
                            yaw, pitch, roll = box.orientation.yaw_pitch_roll
                            total_label_rows.append([sample['frame_index'], roi_index, class_string_to_id(box.name),
                                                     box.center[0], box.center[1], box.center[2],
                                                     box.wlh[0], box.wlh[1], box.wlh[2],
                                                     roll, pitch, yaw])
                            roi_index += 1
                    elif 'CAM' in sensor_type:
                        row = []
                        # timestamp_start
                        row.append(math.floor(prev_timestamp / 1000))
                        # timestamp_stop
                        row.append(math.floor(sample['timestamp'] / 1000))
                        row.append(sampling_time)
                        row.append(sample['filename'])
                        row.append('')
                        data_descriptor_writer.writerow(column for column in row)
                    elif 'LIDAR' in sensor_type:
                        row = []
                        # timestamp_start
                        row.append(math.floor(prev_timestamp / 1000))
                        # timestamp_stop
                        row.append(math.floor(sample['timestamp'] / 1000))
                        row.append(sampling_time)
                        row.append(sample['filename'] + '.ply')
                        data_descriptor_writer.writerow(column for column in row)
                    elif 'RADAR' in sensor_type:
                        row = []
                        # timestamp_start
                        row.append(math.floor(prev_timestamp / 1000))
                        # timestamp_stop
                        row.append(math.floor(sample['timestamp'] / 1000))
                        row.append(sampling_time)
                        row.append(sample['filename'] + '.ply')
                        data_descriptor_writer.writerow(column for column in row)
                    elif 'IMU' in sensor_type:
                        row = []
                        # timestamp_start
                        row.append(math.floor(prev_timestamp / 1000))
                        # timestamp_stop
                        row.append(math.floor(sample['timestamp'] / 1000))
                        row.append(sampling_time)

                        row.append(sample['linear_accel'][0])
                        row.append(sample['linear_accel'][1])
                        row.append(sample['linear_accel'][2])

                        row.append(sample['rotation_rate'][0])
                        row.append(sample['rotation_rate'][1])
                        row.append(sample['rotation_rate'][2])

                        roll = math.atan2(2.0 * (sample['q'][1] * sample['q'][2] + sample['q'][0] * sample['q'][3]), sample['q'][0] * sample['q'][0] + sample['q'][1] * sample['q'][1] - sample['q'][2] * sample['q'][2] - sample['q'][3] * sample['q'][3])
                        pitch = math.asin(-2.0 * (sample['q'][1] * sample['q'][3] - sample['q'][0] * sample['q'][2]))
                        yaw = math.atan2(2.0 * (sample['q'][2] * sample['q'][3]) + sample['q'][0] * sample['q'][1], sample['q'][0] * sample['q'][0] - sample['q'][1] * sample['q'][1] - sample['q'][2] * sample['q'][2] + sample['q'][3] * sample['q'][3])

                        row.append(pitch)
                        row.append(yaw)
                        row.append(roll)

                        data_descriptor_writer.writerow(column for column in row)
                    else:
                        row = []
                        # timestamp_start
                        row.append(math.floor(prev_timestamp / 1000))
                        # timestamp_stop

                        sample_annotation_path = 'C:/dev/NUSCENES/v1.0-mini/sample_annotation.json'
                        sample_data_path = 'C:/dev/NUSCENES/v1.0-mini/sample_data.json'
                        calibrated_sensor_path = 'C:/dev/NUSCENES/v1.0-mini/calibrated_sensor.json'

                        filename = get_sample_filename_by_timestamp(sample_data, sample['timestamp'])
                        if 'CAM_FRONT' == filename.split('/')[1]:
                            row.append(math.floor(sample['timestamp'] / 1000))
                            row.append(sampling_time)
                            row.append(sample['x'])
                            row.append(sample['y'])
                            row.append(sample['velocity'])
                            row.append(sample['yaw'])
                            data_descriptor_writer.writerow(column for column in row)
                    prev_timestamp = sample['timestamp']
                with open(path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(idx + 1) +
                          '/framebased_data_descriptor.csv',
                          mode='w+', newline='') as frame_based_descriptor:
                    frame_based_descriptor_writer = csv.writer(frame_based_descriptor, delimiter=',')
                    frame_based_descriptor_writer.writerow(col for col in frame_based_sync_header)
                    frame_based_descriptor_writer.writerows(total_label_rows)
                # current timestamp will be the previous timestamp in the next iteration


def add_timestamp(sample, sync_list, sensor):
    # search a timestamp in the first column of the list
    for row in sync_list[1:]:
        # if timestamps match
        if int(row[0]) == math.floor(sample['timestamp'] / 1000):
            # replace the -1 with the actual timestamp
            row[sensor+1] = math.floor(sample['timestamp'] / 1000)
            # stop searching if match found
            break


def get_timestamp_interval(list_sensors):
    first_timestamps = []
    last_timestamps = []
    for it in range(len(list_sensors)):
        first_timestamps.append(int(list_sensors[it][0]['timestamp']))
        last_timestamps.append(int(list_sensors[it][-1]['timestamp']))
    return math.floor(min(first_timestamps) / 1000), math.floor(max(last_timestamps) / 1000)


def generate_sync_descriptor(list_sensors, scene_name):
    with open(path_to_destination_folder + '/' + scene_name + 'CyC_sampling_timestamp_sync_temp.csv',
              mode='a',
              newline='') as data_descriptor_file:
        data_descriptor_writer = csv.writer(data_descriptor_file, delimiter=',')

        # write header
        header = []
        header.append('timestamp_stop')
        for it in range(len(SENSORS)):
            header.append('datastream_1' + '_' + str(it + 1))

        data_descriptor_writer.writerow([column for column in header])

        # determine the range of the timestamps
        first_timestamp, last_timestamp = get_timestamp_interval(list_sensors)

        # fill out sync file with timestamps every 1 ms and -1 in all datastreams
        for timestamp in range(first_timestamp, last_timestamp + 1):
            row = []
            row.append(timestamp)
            for it in range(len(SENSORS)):
                row.append(-1)
            data_descriptor_writer.writerow(column for column in row)

    # read the sync file and convert it to a list (for easy fill out with timestamps)
    r = open(path_to_destination_folder + '/' + scene_name + '/CyC_sampling_timestamp_sync_temp.csv', mode='r',
             newline='')
    reader = csv.reader(r)
    sync_list = list(reader)
    r.close()

    # fill out sync file with timestamps of each datastream
    for idx, stream in enumerate(list_sensors):
        for sample in stream:
            add_timestamp(sample, sync_list, idx)

    w = open(path_to_destination_folder + '/' + scene_name + '/sampling_timestamps_sync.csv', mode='w',
             newline='')
    writer = csv.writer(w)
    writer.writerows(sync_list)
    w.close()

    # remove the temporary csv sync file
    os.remove(path_to_destination_folder + '/' + scene_name + '/CyC_sampling_timestamp_sync_temp.csv')


def main():
    with open(scene_file_path) as json_scene_file,\
            open(sample_file_path) as json_sample_file,\
            open(sample_data_file_path) as json_sample_data_file,\
            open(ego_pose_file_path) as json_ego_pose_file,\
            open(instance_file_path) as json_instances_file, \
            open(category_file_path) as json_categories_file, \
            open(annotations_file_path) as json_annotations_file, \
            open(calibration_file_path) as json_calibration_file:
        scene_data = json.load(json_scene_file)
        sample_summary_data = json.load(json_sample_file)
        sample_all_data = json.load(json_sample_data_file)
        ego_pose_data = json.load(json_ego_pose_file)
        annotations_data = json.load(json_annotations_file)
        instances_data = json.load(json_instances_file)
        categories_data = json.load(json_categories_file)
        calibration_data = json.load(json_calibration_file)

        # Set token dependencies between tables
        for record in annotations_data:
            inst = find_sample(instances_data, record['instance_token'])
            record['category_name'] = find_sample(categories_data, inst['category_token'])['name']

        for ann_record in annotations_data:
            sample_record = find_sample(sample_summary_data, ann_record['sample_token'])
            sample_record['anns'] = []
            sample_record['anns'].append(ann_record['token'])

        # for each available scene
        for scene in scene_data:
            # create folder of the current scene
            makedir(path_to_destination_folder, scene['name'])

            # reset current sensor index
            global SENSOR_ITERATOR
            SENSOR_ITERATOR = 0

            list_sensors = list()

            # create folder structure for each sensor
            # add each sensor to a list
            for sensor in range(len(SENSORS)):
                list_sensors.append(make_sensor(path_to_destination_folder, scene['name']))

            print("Processing scene " + scene['name'] + ". Total number of samples is " + str(scene['nbr_samples']))

            current_scene_token = scene['token']
            current_sample_token = scene['first_sample_token']

            # get IMU data for the scene
            IMU_data = nusc_can.get_messages(scene_name=scene['name'], message_name='ms_imu')
            process_scene(list_sensors, IMU_data)

            # loop through all samples
            sample_iterator = 0
            while True:
                sample_iterator += 1
                sample = find_sample(sample_summary_data, current_sample_token)
                current_sample_data = get_sample_data(sample_all_data, current_sample_token)
                for sensor_data in current_sample_data:
                    current_sensor_token = sensor_data['token']
                    prev_sensor = find_sample(sample_all_data, current_sensor_token)
                    prev_sensor_ego_pose = find_sample(ego_pose_data, prev_sensor['ego_pose_token'])
                    frame_id_index = 0
                    while True:
                        sensor = find_sample(sample_all_data, current_sensor_token)
                        sensor_ego_pose = find_sample(ego_pose_data, sensor['ego_pose_token'])
                        annotations = nusc.get_sample_data(sensor['token'])

                        process_sample(path_to_destination_folder, scene['name'], prev_sensor, sensor,
                                       prev_sensor_ego_pose, sensor_ego_pose, list_sensors, annotations, frame_id_index)
                        current_sensor_token = sensor['next']

                        prev_sensor = sensor
                        prev_sensor_ego_pose = find_sample(ego_pose_data, prev_sensor['ego_pose_token'])

                        if current_sensor_token is '':
                            break
                        frame_id_index += 1

                current_sample_token = sample['next']
                if sample_iterator == scene['nbr_samples']:
                    break
                break

            generate_blockchain_descriptor(list_sensors, scene['name'])
            generate_data_descriptor(list_sensors, scene['name'], annotations_data=annotations_data, sample_data=sample_all_data, calibration_data=calibration_data)
            generate_sync_descriptor(list_sensors, scene['name'])


if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot=path_to_database_folder)
    nusc_can = NuScenesCanBus(dataroot=path_to_database_folder)

    main()
