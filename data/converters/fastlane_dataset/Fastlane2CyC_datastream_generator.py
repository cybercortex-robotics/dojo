"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import csv
import json
import math
import os
from shutil import copy

# path to the dataset root
path_to_database_folder = "D:/Datasets/Robinos_data/un_sorted_for_LSTM/"

# add recorded runs with the fastlane car
#runs = {'RFL_ER-CT_31_20170705_154551_split_000', 'RFL_ER-CT_31_20170705_154727_split_000', 'RFL_ER-CT_31_20170705_154853_split_000'}
runs = {'RFL_ER-CT_31_20180201_165107_split_000'}

# path to generated datastream
path_to_destination_folder = "D:/Datasets/fastlane/"

# could be needed for only taking the first frame
bCopy_only_key_frame = False  # save sensory data every 500 [ms]

# for fastlane dataset
# state = x, y, velocity, yaw angle
SENSORS = {'CAM_FRONT': 2, 'GRID': 3, 'STATE': 7}

# create directory
def makedir(dir_path, dir_name):
    if os.path.isdir(dir_path + dir_name):
        print("Folder " + dir_path + dir_name + " already exist. Skipped creation of folder.")
    else:
        try:
            os.makedirs(dir_path + dir_name)
        except OSError:
            pass


# create folder structure for a given sensor type
def make_sensor_folder_structure(path_to_destination_folder, run_name):
    for key, value in SENSORS.items():
        if(key == 'CAM_FRONT'):
            makedir(
                path_to_destination_folder + '/' + run_name + '/datastream_1' + '_' + str(value) + '/', 'samples/')
            makedir(
                path_to_destination_folder + '/' + run_name + '/datastream_1' + '_' + str(value) + '/samples/' + '0' + '/', 'left')
            makedir(
                path_to_destination_folder + '/' + run_name + '/datastream_1' + '_' + str(value) + '/samples/' + '0' + '/', 'right')
        if(key == 'GRID'):
            makedir(
                path_to_destination_folder + '/' + run_name, '/datastream_1' + '_' + str(value) + '/' + 'samples/')
            makedir(
                path_to_destination_folder + '/' + run_name + '/datastream_1' + '_' + str(value) + '/samples/' + '0' + '/', 'left')
            makedir(
                path_to_destination_folder + '/' + run_name + '/datastream_1' + '_' + str(value) + '/samples/' + '0' + '/', 'right')
        if(key is 'STATE'):
            makedir(
                path_to_destination_folder + '/' + run_name, '/datastream_1' + '_' + str(value))


def generate_data_descriptor_for_cam_dataset(list_of_timestamps, list_of_cam_images, path_to_destination_folder, run):
    with open(path_to_destination_folder + '/' + run + '/datastream_1' + '_' + str(SENSORS['CAM_FRONT']) +
              '/data_descriptor.csv',
              mode='a',
              newline='') as data_descriptor_file:
        data_descriptor_writer = csv.writer(data_descriptor_file, delimiter=',')

        sampling_time = 0

        # create header for camera data structure
        header = []
        header.append('timestamp_start')
        header.append('timestamp_stop')
        header.append('sampling_time')
        header.append('left_file_path_0')
        header.append('right_file_path_0')

        data_descriptor_writer.writerow([column for column in header])
        
        prev_timestamp = list_of_timestamps[0]
        
        for idx in range(1,len(list_of_timestamps)):
            sampling_time = int((int(list_of_timestamps[idx]) - int(prev_timestamp)) / 1000)
    
            row = []
            # timestamp_start
            row.append(math.floor(int(prev_timestamp)))
            # timestamp_stop
            row.append(int(list_of_timestamps[idx]))
            row.append(sampling_time)
            row.append("samples/0/left/" + list_of_cam_images[idx])
            row.append('')
            data_descriptor_writer.writerow(column for column in row)

            # current timestamp becomes the previous timestamp
            prev_timestamp = list_of_timestamps[idx]


def generate_data_descriptor_for_grid_dataset(list_of_timestamps, list_of_grid_images, path_to_destination_folder, run):
    with open(path_to_destination_folder + '/' + run + '/datastream_1' + '_' + str(SENSORS['GRID']) +
              '/data_descriptor.csv',
              mode='a',
              newline='') as data_descriptor_file:
        data_descriptor_writer = csv.writer(data_descriptor_file, delimiter=',')
        
        sampling_time = 0
        
        # create header for camera data structure
        header = []
        header.append('timestamp_start')
        header.append('timestamp_stop')
        header.append('sampling_time')
        header.append('left_file_path_0')
        header.append('right_file_path_0')
        
        data_descriptor_writer.writerow([column for column in header])
        
        prev_timestamp = list_of_timestamps[0]
        
        for idx in range(1, len(list_of_timestamps)):
            sampling_time = int((int(list_of_timestamps[idx]) - int(prev_timestamp)) / 1000)
            
            row = []
            # timestamp_start
            row.append(math.floor(int(prev_timestamp)))
            # timestamp_stop
            row.append(int(list_of_timestamps[idx]))
            row.append(sampling_time)
            row.append("samples/0/left/" + list_of_grid_images[idx])
            row.append('')
            data_descriptor_writer.writerow(column for column in row)
            
            # current timestamp becomes the previous timestamp
            prev_timestamp = list_of_timestamps[idx]


def generate_data_descriptor_for_state_dataset(list_of_timestamps, list_of_ego_data_entries, path_to_destination_folder, run):
    with open(path_to_destination_folder + '/' + run + '/datastream_1' + '_' + str(SENSORS['STATE']) +
              '/data_descriptor.csv',
              mode='a',
              newline='') as data_descriptor_file:
        data_descriptor_writer = csv.writer(data_descriptor_file, delimiter=',')
        
        sampling_time = 0
        
        # create header for camera data structure
        header = []
        header.append('timestamp_start')
        header.append('timestamp_stop')
        header.append('sampling_time')
        header.append('state_variable_0')
        header.append('state_variable_1')
        header.append('state_variable_2')
        header.append('state_variable_3')
        
        data_descriptor_writer.writerow([column for column in header])
        
        prev_timestamp = list_of_timestamps[0]
        
        for idx in range(1, len(list_of_timestamps)):
            sampling_time = int((int(list_of_timestamps[idx]) - int(prev_timestamp)) / 1000)
            
            row = []

            # timestamp_start
            row.append(math.floor(int(prev_timestamp)))
            
            # timestamp_stop
            row.append(int(list_of_timestamps[idx]))
            row.append(sampling_time)
            row.append(list_of_ego_data_entries[idx][1]) # x pos in WCS <-> [0] is the timestamp
            row.append(list_of_ego_data_entries[idx][2]) # y pos in WCS
            row.append(list_of_ego_data_entries[idx][3]) # velocity
            row.append(list_of_ego_data_entries[idx][4]) # yaw
            data_descriptor_writer.writerow(column for column in row)

            # current timestamp becomes the previous timestamp
            prev_timestamp = list_of_timestamps[idx]

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


def generate_sync_descriptor(list_of_timestamps, run):
    with open(path_to_destination_folder + '/' + run + '/sampling_timestamps_sync.csv', mode='w',
         newline='') as data_descriptor_file:
        data_descriptor_writer = csv.writer(data_descriptor_file, delimiter=',')

        # write header
        header = []
        header.append('timestamp_stop')
        for key, value in SENSORS.items():
            header.append('datastream_1' + '_' + str(value))

        data_descriptor_writer.writerow([column for column in header])

        # fill out sync file with timestamps every 1 ms and -1 in all datastreams
        for idx in range(0, len(list_of_timestamps)):
            row = []
            row.append(list_of_timestamps[idx])
            for it in range(len(SENSORS)):
                row.append(list_of_timestamps[idx])
            data_descriptor_writer.writerow(column for column in row)
    

def sort_func(e):
    return int(e.split("_")[1].split(".")[0])

    
def get_fastlane_dataset_from_folder(path_to_folder, list_of_camera_images, list_of_grid_images, list_of_ego_data_csv_files):
    # Get the list of all files and directories in the root directory
    dir_list = os.listdir(path_to_folder)
    
    # print the list
    print("Files in '", path_to_folder, "' :")
    print(dir_list)
    print("Nr. of items in folder ", path_to_folder, ":", len(dir_list))

    list_of_camera_images = [f for f in dir_list if f.endswith(".png") is True and f.__contains__("video_image")]
    list_of_grid_images = [f for f in dir_list if f.endswith(".png") is True and f.__contains__("grid_image")]
    #file_list.sort(key=sort_func)


def get_fastlane_camera_images_from_folder(path_to_folder):
    # Get the list of all files and directories in the root directory
    dir_list = os.listdir(path_to_folder)
    
    # print the list
    print("Files in '", path_to_folder, "' :")
    print(dir_list)
    print("Nr. of items in folder ", path_to_folder, ":", len(dir_list))
    
    list_of_camera_images = [f for f in dir_list if f.endswith(".png") is True and f.__contains__("video_image")]
    # file_list.sort(key=sort_func)
    print("Nr. of cam images: ", len(list_of_camera_images))
    return list_of_camera_images


def get_fastlane_grid_images_from_folder(path_to_folder):
    dir_list = os.listdir(path_to_folder)
    list_of_grid_images = [f for f in dir_list if f.endswith(".png") is True and f.__contains__("grid_image")]
    print("Nr. of grid images: ", len(list_of_grid_images))
    return list_of_grid_images


def get_fastlane_csv_files_from_folder(path_to_folder):
    dir_list = os.listdir(path_to_folder)
    list_of_ego_data_csv_files = [f for f in dir_list if f.endswith(".csv") is True]
    print("Nr. of ego data csv files: ", len(list_of_ego_data_csv_files))
    return list_of_ego_data_csv_files


def get_fastlane_timestamps(list_of_camera_images):
    list_of_timestamps = list()
    
    for cam_image_file in list_of_camera_images:
        list_of_timestamps.append(cam_image_file.split('_')[0])
        
    return list_of_timestamps


def check_timestamps_integrity(list_of_timestamps, list_of_camera_images, list_of_grid_images, list_of_ego_data_csv_files):
    for index in range(0, len(list_of_timestamps)):
        if list_of_timestamps[index] != list_of_camera_images[index].split('_')[0]:
            print("The index at which camera images do not match is ", index)
            return False
        if list_of_timestamps[index] != list_of_grid_images[index].split('_')[0]:
            print("The index at which grid images do not match is ", index)
            return False
        if list_of_timestamps[index] != list_of_ego_data_csv_files[index].split('_')[0]:
            print("The index at which ego_data does not match is ", index)
            return False
    return True


def copy_cam_images(list_of_cam_images, run):
    for idx in range (0, len(list_of_cam_images)):
        src = path_to_database_folder + run + "/" + list_of_cam_images[idx]
        dst = path_to_destination_folder + '/' + run + '/datastream_1' + '_' + str(SENSORS['CAM_FRONT']) + '/' \
              + 'samples/0/left/' + list_of_cam_images[idx]
        copy(src, dst)
    
    
def copy_grid_images(list_of_grid_images, run):
    for idx in range (0, len(list_of_grid_images)):
        src = path_to_database_folder + run + "/" + list_of_grid_images[idx]
        dst = path_to_destination_folder + '/' + run + '/datastream_1' + '_' + str(SENSORS['GRID']) + '/' \
              + "samples/0/left/" + list_of_grid_images[idx]
        copy(src, dst)
        

def read_fastlane_csv_files(path_to_database_folder, run, list_of_ego_data_csv_files):
    list_of_ego_data_entries = list()
    
    for idx in range(0, len(list_of_ego_data_csv_files)):
        csvfile = open(path_to_database_folder + '/' + run + '/' + list_of_ego_data_csv_files[idx])
        readCSV = csv.reader(csvfile, delimiter=';')
        
        # need to skip the first line which contains the headers
        next(readCSV)
        
        ego_data_entry = list()
        for row in readCSV:
            timestamp = row[0]
            posX = row[15]
            posY = row[16]
            yaw = row[18]
            speedX = row[23]
            speedY = row[24]

            # compute speed
            velocity = math.sqrt(float(speedX) * float(speedX) + float(speedY) * float(speedY))
            
            #print(timestamp, posX, posY, yaw, velocity)

            ego_data_entry.append(timestamp)
            ego_data_entry.append(posX)
            ego_data_entry.append(posY)
            ego_data_entry.append(velocity)
            ego_data_entry.append(yaw)
            
        # append each read line to the list of ego data entries
        list_of_ego_data_entries.append(ego_data_entry)
        
    return list_of_ego_data_entries
            
def main():
    # for each run
    for run in runs:
        # create folder of the current run
        makedir(path_to_destination_folder, run)
        
        # create folder structure for each sensor
        for sensor in range(len(SENSORS)):
            make_sensor_folder_structure(path_to_destination_folder, run)

        print("Processing run " + run + ".")
        #Total number of samples is " + str(scene['nbr_samples']));

        list_of_camera_images = list()
        list_of_grid_images = list()
        list_of_ego_data_csv_files = list()
        
        list_of_camera_images = get_fastlane_camera_images_from_folder(path_to_database_folder + run)
        list_of_grid_images = get_fastlane_grid_images_from_folder(path_to_database_folder + run)
        list_of_ego_data_csv_files = get_fastlane_csv_files_from_folder(path_to_database_folder + run)
        
        # get timestamps from item names
        list_of_timestamps = list()
        list_of_timestamps = get_fastlane_timestamps(list_of_camera_images)
        
        # check that the timestamps match for camera, grid, and csv file
        if check_timestamps_integrity(list_of_timestamps, list_of_camera_images, list_of_grid_images, list_of_ego_data_csv_files) is False:
            print("The timestamps of camera images, grid images, and ego data csv files do NOT match! Exiting...")
        
        # generate data descriptors
        generate_data_descriptor_for_cam_dataset(list_of_timestamps, list_of_camera_images, path_to_destination_folder, run)
        generate_data_descriptor_for_grid_dataset(list_of_timestamps, list_of_grid_images, path_to_destination_folder, run)
        
        #copy data
        copy_cam_images(list_of_camera_images, run)
        copy_grid_images(list_of_grid_images, run)
        
        #create ego data
        list_of_ego_data_entries = list()
        list_of_ego_data_entries = read_fastlane_csv_files(path_to_database_folder, run, list_of_ego_data_csv_files)

        generate_data_descriptor_for_state_dataset(list_of_timestamps, list_of_ego_data_entries, path_to_destination_folder, run)
        
        # generate sync descriptor (sampling_timestamps_sync)
        generate_sync_descriptor(list_of_timestamps, run)
        
        quit()
        
    quit()



if __name__ == "__main__":
    main();
