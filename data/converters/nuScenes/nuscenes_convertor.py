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

import os
import csv
import math
import shutil
from pathlib import Path
from copy import deepcopy
import json

import numpy as np
from shutil import copy
from nuscenes.utils.geometry_utils import box_in_image

from box_utils import view_points, get_visibility_token
from cloud_utils import load_lidar_from_file, write_lidar_ply, load_radar_from_file, write_radar_ply
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
#from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from data import types_CyC_TYPES

path_to_database_folder = r"D:/dev/NUSCENES/"
path_to_destination_folder = r"D:/NuScenes/"

bCopy_only_key_frame = False  # save sensory data every 500 [ms]

REPAIR = True

SENSORS = {'CAM_FRONT': 0, 'CAM_FRONT_LEFT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK': 3, 'CAM_BACK_LEFT': 4, 'CAM_BACK_RIGHT': 5,
           'LIDAR_TOP': 6,
           'RADAR_FRONT': 7, 'RADAR_FRONT_LEFT': 8, 'RADAR_FRONT_RIGHT': 9, 'RADAR_BACK_LEFT': 10, 'RADAR_BACK_RIGHT': 11,
           'VehicleOdometry': 12,
           '2D_CAM_FRONT': 13, '2D_CAM_FRONT_LEFT': 14, '2D_CAM_FRONT_RIGHT': 15, '2D_CAM_BACK': 16, '2D_CAM_BACK_LEFT': 17, '2D_CAM_BACK_RIGHT': 18,
           '3D_CAM_FRONT': 19, '3D_CAM_FRONT_LEFT': 20, '3D_CAM_FRONT_RIGHT': 21, '3D_CAM_BACK': 22, '3D_CAM_BACK_LEFT': 23, '3D_CAM_BACK_RIGHT': 24, '3D_LIDAR_TOP': 25}


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
        'animal': 2,
        'human.pedestrian.wheelchair': 1,
        'human.pedestrian.adult': 1,
        'human.pedestrian.child': 2,
        'human.pedestrian.stroller': 2,
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
        'vehicle.emergency.ambulance': 14,
        'vehicle.emergency.police': 14,
        'vehicle.truck': 18
    }

    return class_dict[class_string]

cursed_classes = [
    'static_object.bicycle_rack',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer'
]


# create bat file
def create_batch_file(database_path, scene_name):
    """
        Takes the conda environment from which the script is runned.
        Writes the batch file with respect to the current environment, so must have all needed packages.
    """
    full_path = Path(database_path + '/' + scene_name)
    with open(full_path / 'run_viz.bat', 'w+') as f:
        viz_script_path = '%CyC_DIR%' + '/data/viz/MainWindow.py'
        environment = 'python'
        dataset_dir = database_path + scene_name
        f.write(environment + ' {} {}'.format(viz_script_path, dataset_dir))
        f.close()

# create directory
def makedir(dir_path, dir_name):
    if os.path.isdir(dir_path + dir_name):
        print("Folder " + dir_path + dir_name + " already exist. Skip command.")
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
def make_sensor(path_to_destination_folder, scene_name, sensor):
    if not CAMERA_SENSOR[SENSORS[sensor]]:
        makedir(
            path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                SENSORS[sensor]+1) + '/', 'samples/')
    else:
        makedir(
            path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                SENSORS[sensor]+1) + '/', 'samples/')
        makedir(
            path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                SENSORS[sensor]+1) + '/samples/', 'left')
        makedir(
            path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                SENSORS[sensor]+1) + '/samples/', 'right')
    return list()

def get_state(prev_sensor, sensor, prev_sensor_ego_pose, sensor_ego_pose):
    yaw = Quaternion(sensor_ego_pose['rotation']).yaw_pitch_roll[0]
    time_delta = (1e-6 * sensor['timestamp']) - (1e-6 * prev_sensor['timestamp'])  # 1e-6 is for converting from uSec to sec
    pose_delta_x = sensor_ego_pose['translation'][0] - prev_sensor_ego_pose['translation'][0]
    pose_delta_y = sensor_ego_pose['translation'][1] - prev_sensor_ego_pose['translation'][1]

    velocity_overall = 0
    if time_delta > 0:
        velocity_x = pose_delta_x / time_delta
        velocity_y = pose_delta_y / time_delta
        velocity_overall = math.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)

    return {'timestamp': sensor['timestamp'], 'x': sensor_ego_pose['translation'][0],
            'y': sensor_ego_pose['translation'][1], 'z': sensor_ego_pose['translation'][2], 'velocity': velocity_overall,
            'yaw': yaw}


def process_scene(list_sensors, IMU_data):
    if len(IMU_data) > 0:
        for imu_data in IMU_data:
            old_keyname = 'utime'
            new_keyname = 'timestamp'
            imu_data[new_keyname] = imu_data.pop(old_keyname)
            list_sensors[SENSORS['IMU']].append(imu_data)


# copy sample file and add the information to a list for further processing
def process_sample(path_to_destination_folder, scene_name,
                   list_sensors, sensor_data, sensor_annotations, sensor_state,
                   sensor_records, frame_idx, sensor_channel, sample_annotations, REPAIR=False):
    path, filename = os.path.split(sensor_data['filename'])

    if sensor_channel in SENSORS:
        # append sample data to the list
        list_sensors[SENSORS[sensor_channel]].append({'timestamp': sensor_data['timestamp'],
                                                      'filename': 'samples/' + 'left/' + filename if CAMERA_SENSOR[
                                                   SENSORS[sensor_channel]] else 'samples/' + filename})

        if 'CAM' in sensor_channel:
            if '2D_'+sensor_channel in SENSORS:
                box_2d_entries = []
                for box in sensor_annotations:
                    aux_box = deepcopy(box)
                    aux_box.translate(-np.array(sensor_records['calib_translation']))
                    aux_box.rotate(Quaternion(sensor_records['calib_rotation']).inverse)
                    corners = view_points(points=aux_box.corners(wlh_factor=0.9), view=sensor_records['camera_params']['camera_intrinsic'], normalize=True)[:2, :]
                    corners = np.transpose(corners)
                    xs_ = sorted(corners[:, 0])
                    ys_ = sorted(corners[:, 1])

                    # if box.name in cursed_classes:
                    #     x1 = int(np.sum(xs_[0:3:2]) / 2)
                    #     y1 = int(np.sum(ys_[0:3:2]) / 2)
                    #     x2 = int(np.sum(xs_[5::2]) / 2)
                    #     y2 = int(np.sum(ys_[5::2]) / 2)
                    # else:
                    #     x1 = xs_[1]
                    #     x2 = xs_[-2]
                    #     y1 = ys_[1]
                    #     y2 = ys_[-2]

                    x1 = int(np.sum(xs_[0:2]) / 2)
                    y1 = int(np.sum(ys_[0:2]) / 2)
                    x2 = int(np.sum(xs_[6:]) / 2)
                    y2 = int(np.sum(ys_[6:]) / 2)

                    x1_cropped, x2_cropped, y1_cropped, y2_cropped = x1, x2, y1, y2

                    if x1 < 0:
                        x1_cropped = 0
                    elif x1 > 1599:
                        x1_cropped = 1599

                    if y1 < 0:
                        y1_cropped = 0
                    elif y1 > 899:
                        y1_cropped = 899

                    if x2 < 0:
                        x2_cropped = 0
                    elif x2 > 1599:
                        x2_cropped = 1599

                    if y2 < 0:
                        y2_cropped = 0
                    elif y2 > 899:
                        y2_cropped = 899

                    w, h = (x2 - x1), (y2 - y1)
                    w_cropped, h_cropped = (x2_cropped - x1_cropped), (y2_cropped - y1_cropped)

                    area = w * h
                    area_cropped = w_cropped * h_cropped

                    crop_factor = area_cropped / area
                    box.set_crop_factor(crop_factor)

                    if int(get_visibility_token(sample_annotations, box.token)) > 1 and box.crop_factor > 0.225:
                        box_2d_entries.append((aux_box.name, x1_cropped, y1_cropped, w_cropped, h_cropped))
                list_sensors[SENSORS['2D_' + sensor_channel]].append({
                    'timestamp': sensor_data['timestamp'],
                    'channel': '2D_' + sensor_channel,
                    'frame_index': frame_idx,
                    'boxes': box_2d_entries,
                    })
            if '3D_'+sensor_channel in SENSORS:
                list_sensors[SENSORS['3D_' + sensor_channel]].append(
                    {'timestamp': sensor_data['timestamp'],
                     'channel': '3D_' + sensor_channel,
                     'frame_index': frame_idx,
                     'boxes': [box for box in sensor_annotations if (int(get_visibility_token(sample_annotations, box.token)) > 1) and box.crop_factor > 0.225]})
        elif 'LIDAR' in sensor_channel:
            list_sensors[SENSORS['3D_' + sensor_channel]].append({
                'timestamp': sensor_data['timestamp'],
                'channel': '3D_' + sensor_channel,
                'frame_index': frame_idx,
                'boxes': [box for box in sensor_annotations]})
        # append the state only for the camera front sensor
        if sensor_channel == 'CAM_FRONT':
            list_sensors[SENSORS['VehicleOdometry']].append(sensor_state)

            src = path_to_database_folder + sensor_data['filename']

            dst = path_to_destination_folder + '/'  + scene_name + '/datastream_1_' + str(SENSORS[sensor_channel] + 1) + \
                  '/samples/left/'
            if not REPAIR:
                copy(src, dst)
        elif sensor_channel == 'CAM_FRONT_LEFT':
            src = path_to_database_folder + sensor_data['filename']
            dst = path_to_destination_folder + '/' + scene_name + '/datastream_1_' + str(SENSORS[sensor_channel] + 1) + \
                  '/samples/left/'
            if not REPAIR:
                copy(src, dst)
        elif sensor_channel == 'CAM_FRONT_RIGHT':
            src = path_to_database_folder + sensor_data['filename']
            dst = path_to_destination_folder + '/' + scene_name + '/datastream_1_' + str(SENSORS[sensor_channel] + 1) + \
                  '/samples/left/'
            if not REPAIR:
                copy(src, dst)
        elif sensor_channel == 'CAM_BACK':
            src = path_to_database_folder + sensor_data['filename']
            dst = path_to_destination_folder + '/' + scene_name + '/datastream_1_' + str(SENSORS[sensor_channel] + 1) + \
                  '/samples/left/'
            if not REPAIR:
                copy(src, dst)
        elif sensor_channel == 'CAM_BACK_LEFT':
            src = path_to_database_folder + sensor_data['filename']
            dst = path_to_destination_folder + '/' + scene_name + '/datastream_1_' + str(SENSORS[sensor_channel] + 1) + \
                  '/samples/left/'
            if not REPAIR:
                copy(src, dst)
        elif sensor_channel == 'CAM_BACK_RIGHT':
            src = path_to_database_folder + sensor_data['filename']
            dst = path_to_destination_folder + '/' + scene_name + '/datastream_1_' + str(SENSORS[sensor_channel] + 1) + \
                  '/samples/left/'
            if not REPAIR:
                copy(src, dst)
        elif 'LIDAR' in sensor_channel:
            # load lidar bin file
            lidar_coud = load_lidar_from_file(path_to_database_folder + sensor_data['filename'])
            file_path = Path(path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                    SENSORS[sensor_channel] + 1) + '/samples/' + sensor_data['filename'].split('/')[-1] + '.ply')

            dirname = os.path.dirname(file_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            if not REPAIR:
                write_lidar_ply(file_path, lidar_coud)
        elif 'RADAR' in sensor_channel:
            radar_cloud = load_radar_from_file(path_to_database_folder + sensor_data['filename'])
            file_path = Path(path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(
                    SENSORS[sensor_channel] + 1) + '/samples/' + sensor_data['filename'].split('/')[-1] + '.ply')

            dirname = os.path.dirname(file_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            if not REPAIR:
                write_radar_ply(file_path, radar_cloud)


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
                      'VehicleOdometry': 'CyC_VEHICLE_STATE_ESTIMATION_FILTER_TYPE',
                      '2D_CAM_FRONT': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '2D_CAM_FRONT_LEFT': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '2D_CAM_FRONT_RIGHT': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '2D_CAM_BACK': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '2D_CAM_BACK_LEFT': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '2D_CAM_BACK_RIGHT': 'CyC_OBJECT_DETECTOR_2D_FILTER_TYPE',
                      '3D_CAM_FRONT': 'CyC_OBJECT_DETECTOR_3D_FILTER_TYPE',
                      '3D_CAM_FRONT_LEFT': 'CyC_BBOX_DETECTOR_3D_FILTER_TYPE',
                      '3D_CAM_FRONT_RIGHT': 'CyC_BBOX_DETECTOR_3D_FILTER_TYPE',
                      '3D_CAM_BACK': 'CyC_BBOX_DETECTOR_3D_FILTER_TYPE',
                      '3D_CAM_BACK_LEFT': 'CyC_BBOX_DETECTOR_3D_FILTER_TYPE',
                      '3D_CAM_BACK_RIGHT': 'CyC_BBOX_DETECTOR_3D_FILTER_TYPE',
                      #'IMU': 'CyC_IMU_FILTER_TYPE'
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
                    #'IMU': 'CyC_IMU'
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
                objdet_row[('2D_' + 'CAM_FRONT_LEFT')].append(str(idx + 14))
                objdet_row[('2D_' + 'CAM_FRONT_LEFT')].append('ObjectDetector2D_'+str('CAM_FRONT_LEFT'))
                objdet_row[('2D_' + 'CAM_FRONT_LEFT')].append(str(filter_type[SENSORS_FILTER['2D_CAM_FRONT_LEFT']]))
                objdet_row[('2D_' + 'CAM_FRONT_LEFT')].append(str(data_type[SENSORS_DATA['2D_CAM_FRONT_LEFT']]))
                objdet_row[('2D_' + 'CAM_FRONT_LEFT')].append("{1-2;1-13}")
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append(str(idx + 20))
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append('ObjectDetector3D_' + str('CAM_FRONT_LEFT'))
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append(str(filter_type[SENSORS_FILTER['3D_CAM_FRONT_LEFT']]))
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append(str(data_type[SENSORS_DATA['3D_CAM_FRONT_LEFT']]))
                objdet_row[('3D_' + 'CAM_FRONT_LEFT')].append("{1-2;1-13}")
            elif 'CAM_FRONT_RIGHT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CameraFrontRight')
                row.append(str(filter_type[SENSORS_FILTER['CAM_FRONT_RIGHT']]))
                row.append(str(data_type[SENSORS_DATA['CAM_FRONT_RIGHT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append(str(vision_core_id))
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append(str(idx + 14))
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append('ObjectDetector2D_' + str('CAM_FRONT_RIGHT'))
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append(str(filter_type[SENSORS_FILTER['2D_CAM_FRONT_RIGHT']]))
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append(str(data_type[SENSORS_DATA['2D_CAM_FRONT_RIGHT']]))
                objdet_row[('2D_' + 'CAM_FRONT_RIGHT')].append("{1-3;1-13}")
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append(str(idx + 20))
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append('ObjectDetector3D_' + str('CAM_FRONT_RIGHT'))
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append(str(filter_type[SENSORS_FILTER['3D_CAM_FRONT_RIGHT']]))
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append(str(data_type[SENSORS_DATA['3D_CAM_FRONT_RIGHT']]))
                objdet_row[('3D_' + 'CAM_FRONT_RIGHT')].append("{1-3;1-13}")
            elif 'CAM_FRONT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CameraFront')
                row.append(str(filter_type[SENSORS_FILTER['CAM_FRONT']]))
                row.append(str(data_type[SENSORS_DATA['CAM_FRONT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                objdet_row[('2D_' + 'CAM_FRONT')].append(str(vision_core_id))
                objdet_row[('2D_' + 'CAM_FRONT')].append(str(idx + 14))
                objdet_row[('2D_' + 'CAM_FRONT')].append('ObjectDetector2D_' + str('CAM_FRONT'))
                objdet_row[('2D_' + 'CAM_FRONT')].append(str(filter_type[SENSORS_FILTER['2D_CAM_FRONT']]))
                objdet_row[('2D_' + 'CAM_FRONT')].append(str(data_type[SENSORS_DATA['2D_CAM_FRONT']]))
                objdet_row[('2D_' + 'CAM_FRONT')].append("{1-1;1-13}")
                objdet_row[('3D_' + 'CAM_FRONT')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_FRONT')].append(str(idx + 20))
                objdet_row[('3D_' + 'CAM_FRONT')].append('ObjectDetector3D_' + str('CAM_FRONT'))
                objdet_row[('3D_' + 'CAM_FRONT')].append(str(filter_type[SENSORS_FILTER['3D_CAM_FRONT']]))
                objdet_row[('3D_' + 'CAM_FRONT')].append(str(data_type[SENSORS_DATA['3D_CAM_FRONT']]))
                objdet_row[('3D_' + 'CAM_FRONT')].append("{1-1;1-13}")
            elif 'CAM_BACK_LEFT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CameraBackLeft')
                row.append(str(filter_type[SENSORS_FILTER['CAM_BACK_LEFT']]))
                row.append(str(data_type[SENSORS_DATA['CAM_BACK_LEFT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append(str(vision_core_id))
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append(str(idx + 14))
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append('ObjectDetector2D_' + str('CAM_BACK_LEFT'))
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append(str(filter_type[SENSORS_FILTER['2D_CAM_BACK_LEFT']]))
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append(str(data_type[SENSORS_DATA['2D_CAM_BACK_LEFT']]))
                objdet_row[('2D_' + 'CAM_BACK_LEFT')].append("{1-5;1-13}")
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append(str(idx + 20))
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append('ObjectDetector3D_' + str('CAM_BACK_LEFT'))
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append(str(filter_type[SENSORS_FILTER['3D_CAM_BACK_LEFT']]))
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append(str(data_type[SENSORS_DATA['3D_CAM_BACK_LEFT']]))
                objdet_row[('3D_' + 'CAM_BACK_LEFT')].append("{1-5;1-13}")
            elif 'CAM_BACK_RIGHT' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CameraBackRight')
                row.append(str(filter_type[SENSORS_FILTER['CAM_BACK_RIGHT']]))
                row.append(str(data_type[SENSORS_DATA['CAM_BACK_RIGHT']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append(str(vision_core_id))
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append(str(idx + 14))
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append('ObjectDetector2D_' + str('CAM_BACK_RIGHT'))
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append(str(filter_type[SENSORS_FILTER['2D_CAM_BACK_RIGHT']]))
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append(str(data_type[SENSORS_DATA['2D_CAM_BACK_RIGHT']]))
                objdet_row[('2D_' + 'CAM_BACK_RIGHT')].append("{1-6;1-13}")
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append(str(idx + 20))
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append('ObjectDetector3D_' + str('CAM_BACK_RIGHT'))
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append(str(filter_type[SENSORS_FILTER['3D_CAM_BACK_RIGHT']]))
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append(str(data_type[SENSORS_DATA['3D_CAM_BACK_RIGHT']]))
                objdet_row[('3D_' + 'CAM_BACK_RIGHT')].append("{1-6;1-13}")
            elif 'CAM_BACK' in sensor_type:
                row = []
                row.append(str(vision_core_id))
                row.append(str(idx + 1))
                row.append('CameraBack')
                row.append(str(filter_type[SENSORS_FILTER['CAM_BACK']]))
                row.append(str(data_type[SENSORS_DATA['CAM_BACK']]))
                blockchain_descriptor_writer.writerow(column for column in row)
                objdet_row[('2D_' + 'CAM_BACK')].append(str(vision_core_id))
                objdet_row[('2D_' + 'CAM_BACK')].append(str(idx + 14))
                objdet_row[('2D_' + 'CAM_BACK')].append('ObjectDetector2D_' + str('CAM_BACK'))
                objdet_row[('2D_' + 'CAM_BACK')].append(str(filter_type[SENSORS_FILTER['2D_CAM_BACK']]))
                objdet_row[('2D_' + 'CAM_BACK')].append(str(data_type[SENSORS_DATA['2D_CAM_BACK']]))
                objdet_row[('2D_' + 'CAM_BACK')].append("{1-4;1-13}")
                objdet_row[('3D_' + 'CAM_BACK')].append(str(vision_core_id))
                objdet_row[('3D_' + 'CAM_BACK')].append(str(idx + 20))
                objdet_row[('3D_' + 'CAM_BACK')].append('ObjectDetector3D_' + str('CAM_BACK'))
                objdet_row[('3D_' + 'CAM_BACK')].append(str(filter_type[SENSORS_FILTER['3D_CAM_BACK']]))
                objdet_row[('3D_' + 'CAM_BACK')].append(str(data_type[SENSORS_DATA['3D_CAM_BACK']]))
                objdet_row[('3D_' + 'CAM_BACK')].append("{1-4;1-13}")
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


def generate_data_descriptor(list_sensors, scene_name):
    for idx, sensor in enumerate(list_sensors):
        if len(sensor) > 0 and not os.path.exists(path_to_destination_folder + '/' + scene_name + '/datastream_1' + '_' + str(idx + 1) +
                      '/data_descriptor.csv'):
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
                if sensor[0]:
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
                        header.append('frame_id')
                        frame_based_sync_header.append('frame_id')
                        frame_based_sync_header.append('roi_id')
                        frame_based_sync_header.append('cls')
                        frame_based_sync_header.append('x')
                        frame_based_sync_header.append('y')
                        frame_based_sync_header.append('width')
                        frame_based_sync_header.append('height')
                    elif '3D' in sensor_type:
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
                                total_label_rows.append([sample['frame_index'], roi_index, class_string_to_id(box[0]),
                                                         box[1], box[2], box[3], box[4]])
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
                                                         box.wlh[0], box.wlh[2], box.wlh[1],
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
    with open(path_to_destination_folder + '/' + scene_name + '/CyC_sampling_timestamp_sync_temp.csv',
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


def get_sensor_records_dict(token: str) -> dict:
    """
    Given the sensor token, get the pose record and calibration record from database.
    :param token: String representing the sensor_token.
    :return: Dictionary containing pose parameters for the specific sensor.
    """
    sd_record = nusc.get('sample_data', token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    records_dict = {'type': sensor_record['modality']}
    if records_dict['type'] == 'camera':
        records_dict['camera_params'] = {'camera_intrinsic': np.array(cs_record['camera_intrinsic']),
                                         'imsize': (sd_record['width'], sd_record['height'])}
    records_dict['pose_translation'] = pose_record['translation']
    records_dict['pose_rotation'] = pose_record['rotation']
    records_dict['calib_translation'] = cs_record['translation']
    records_dict['calib_rotation'] = cs_record['rotation']

    return records_dict


def get_sensor_bounding_boxes(sensor_records_dict: dict, token: str) -> list:
    """
    Given the sensor tokens, apply transformations based on the sensor type.
    :param token: String representing the sensor_token.
    :param sensor_records_dict: Dictionary containing sensor calibration parameters (translation, rotation)
    :return: list of bounding boxes.
    """
    final_boxes_list = []
    bounding_boxes = nusc.get_boxes(token)
    if sensor_records_dict['type'] in ['lidar']:
        for box in bounding_boxes:
            yaw = Quaternion(sensor_records_dict['pose_rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(sensor_records_dict['pose_translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            final_boxes_list.append(box)
    elif sensor_records_dict['type'] == 'camera':
        for box in bounding_boxes:
            aux_box = deepcopy(box)
            aux_box.translate(-np.array(sensor_records_dict['pose_translation']))
            aux_box.rotate(Quaternion(sensor_records_dict['pose_rotation']).inverse)

            #  Move box to sensor coord system.
            aux_box.translate(-np.array(sensor_records_dict['calib_translation']))
            aux_box.rotate(Quaternion(sensor_records_dict['calib_rotation']).inverse)

            if box_in_image(aux_box,
                            sensor_records_dict['camera_params']['camera_intrinsic'],
                            sensor_records_dict['camera_params']['imsize']):
                box.translate(-np.array(sensor_records_dict['pose_translation']))
                box.rotate(Quaternion(sensor_records_dict['pose_rotation']).inverse)

                final_boxes_list.append(box)

    return final_boxes_list


def remove_target_datastreams(scenes):
    detection_datastreams = list(range(14, 26))

    assert os.path.exists(path_to_destination_folder)

    for scene in scenes:
        scene_path = os.path.join(path_to_destination_folder, scene['name'])
        for datastream in detection_datastreams:
            datastream_path = os.path.join(scene_path, 'datastream_1_{}'.format(str(datastream)))
            if os.path.exists(datastream_path):
                shutil.rmtree(datastream_path)
                print("removed " + datastream_path)


def remove_target_blockchain(scenes):
    assert os.path.exists(path_to_destination_folder)

    for scene in scenes:
        scene_path = os.path.join(path_to_destination_folder, scene['name'])
        blockchain_path = os.path.join(scene_path, 'datablock_descriptor.csv')
        if os.path.exists(blockchain_path):
            os.remove(blockchain_path)


def main():
        # Get all scenes
        scenes = nusc.scene
        sample_annotations_file = open(path_to_database_folder + '/v1.0-mini/sample_annotation.json')
        sample_annotations = json.load(sample_annotations_file)

        # Iterate scenes and create folder for each one
        for scene in scenes:
            makedir(path_to_destination_folder, scene['name'])

            # reset current sensor index
            global SENSOR_ITERATOR
            SENSOR_ITERATOR = 0
            list_sensors = list()

            # create folder structure for each sensor
            # add each sensor to a list
            for sensor in SENSORS:
                list_sensors.append(make_sensor(path_to_destination_folder, scene['name'], sensor))

            print("Processing scene " + scene['name'] + ". Total number of samples is " + str(scene['nbr_samples']))

            # get IMU data for the scene
            #IMU_data = nusc_can.get_messages(scene_name=scene['name'], message_name='ms_imu')
            #process_scene(list_sensors, IMU_data)

            # NEW APPROACH
            first_sample_token = scene['first_sample_token']
            current_sample_data = nusc.get('sample', first_sample_token)
            frame_id_index = 0
            while current_sample_data['next'] != '':
                prev_sensor = nusc.get('sample_data', current_sample_data['data']['RADAR_FRONT'])
                prev_sensor_ego_pose = nusc.get('ego_pose', prev_sensor['ego_pose_token'])
                for channel in current_sample_data['data']:
                    sensor_token = current_sample_data['data'][channel]
                    records = get_sensor_records_dict(sensor_token)
                    sensor_annotations = get_sensor_bounding_boxes(records, sensor_token)

                    sensor = nusc.get('sample_data', sensor_token)
                    sensor_ego_pose = nusc.get('ego_pose', sensor['ego_pose_token'])
                    sensor_state = get_state(prev_sensor, sensor, prev_sensor_ego_pose, sensor_ego_pose)

                    process_sample(path_to_destination_folder, scene['name'],
                                   list_sensors, sensor, sensor_annotations,
                                   sensor_state, records, frame_id_index, channel, sample_annotations,
                                   REPAIR=REPAIR)

                frame_id_index += 1
                current_sample_data = nusc.get('sample', current_sample_data['next'])

            if REPAIR is False:
                generate_sync_descriptor(list_sensors, scene['name'])

            generate_blockchain_descriptor(list_sensors, scene['name'])
            generate_data_descriptor(list_sensors, scene['name'])

            # Generate batch file
            if REPAIR is False:
                create_batch_file(path_to_destination_folder, scene['name'])


if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot=path_to_database_folder)
    # nusc_can = NuScenesCanBus(dataroot=path_to_database_folder)
    if REPAIR is True:
        remove_target_datastreams(nusc.scene)
        remove_target_blockchain(nusc.scene)
    main()
