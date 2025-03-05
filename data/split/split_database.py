"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from tqdm import tqdm
import pandas as pd
import shutil
import os


class SplitDatabase:
    def __init__(self, input_dir, output_dir):
        self.input_database_dir = input_dir
        self.output_datasets_dir = output_dir

    @staticmethod
    def clear_directory(directory_path):
        directory_list = os.listdir(directory_path)

        if len(directory_list) == 0:
            return
        else:
            for filename in directory_list:
                file_path = os.path.join(directory_path, filename)

                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
                    exit(1)

    @staticmethod
    def check_csv_is_empty(csv_file_source, csv_file_destination):
        with open(csv_file_source) as csvfile:
            lines = csvfile.readlines()
            if len(lines) <= 1:
                shutil.copyfile(csv_file_source, csv_file_destination)
                return True
        return False

    @staticmethod
    def copy_files(filter_source_dir, filter_destination_dir):
        # Prepare csv files
        data_descriptor_file_path = "data_descriptor.csv"
        data_descriptor_file_source = os.path.join(filter_destination_dir, data_descriptor_file_path)
        data_descriptor_frame = pd.read_csv(data_descriptor_file_source)

        # Copy files
        for column_name in ["voxels_file_path", "radar_file_path_0", "left_file_path_0",
                            "right_file_path_0", "semantic", "instances", "octree_file_path_"]:
            if column_name in data_descriptor_frame.columns:
                for relative_path in data_descriptor_frame[column_name]:
                    if not isinstance(relative_path, str):
                        continue

                    new_source_dir      = os.path.join(     filter_source_dir, relative_path)
                    new_destination_dir = os.path.join(filter_destination_dir, relative_path)
                    shutil.copyfile(new_source_dir, new_destination_dir)

    def prepare_output_directory(self, timestamp_init, timestamp_end):
        self.input_database_dir.replace('/', '\\')
        database_name = self.input_database_dir.split('\\')[-1]
        output_database_dir = os.path.join(self.output_datasets_dir,
                                           "{}_{}_{}".format(database_name, timestamp_init, timestamp_end))
        if not os.path.exists(output_database_dir):
            os.mkdir(output_database_dir)
        else:
            self.clear_directory(output_database_dir)

        return output_database_dir

    def generate_main_csv_files(self, output_directory, timestamp_init, timestamp_end):
        # Copy datablock_descriptor.csv
        datablock_descriptor_filename = "datablock_descriptor.csv"
        datablock_descriptor_source      = os.path.join(self.input_database_dir, datablock_descriptor_filename)
        datablock_descriptor_destination = os.path.join(       output_directory, datablock_descriptor_filename)
        shutil.copyfile(datablock_descriptor_source, datablock_descriptor_destination)

        # Prepare sampling_timestamps_sync.csv
        sampling_timestamps_sync_filename    = "sampling_timestamps_sync.csv"
        sampling_timestamps_sync_source      = os.path.join(self.input_database_dir, sampling_timestamps_sync_filename)
        sampling_timestamps_sync_destination = os.path.join(       output_directory, sampling_timestamps_sync_filename)

        # Extract rows between timestamp_init and timestamp_end
        sampling_timestamps_sync_data_frame = pd.read_csv(sampling_timestamps_sync_source)

        sampling_timestamps_sync_data_frame = \
            sampling_timestamps_sync_data_frame[
                sampling_timestamps_sync_data_frame['timestamp_stop'].between(timestamp_init, timestamp_end)]
        sampling_timestamps_sync_data_frame.to_csv(sampling_timestamps_sync_destination, index=False)

    def generate_framebased_data_file(self, filter_dir_source, filter_dir_destination,
                                      timestamp_init, timestamp_end, file_name):
        # Prepare framebased data csv file
        framebased_data_descriptor_file_source      = os.path.join(     filter_dir_source, file_name)
        framebased_data_descriptor_file_destination = os.path.join(filter_dir_destination, file_name)

        # Check if csv file is empty
        if self.check_csv_is_empty(framebased_data_descriptor_file_source, framebased_data_descriptor_file_destination):
            return

        # Extract rows between timestamp_init and timestamp_end if timestamp_stop colum exists
        framebased_data_descriptor_frame = pd.read_csv(framebased_data_descriptor_file_source)

        if 'timestamp_stop' in framebased_data_descriptor_frame.columns:
            framebased_data_descriptor_frame = framebased_data_descriptor_frame[
                    framebased_data_descriptor_frame['timestamp_stop'].between(timestamp_init, timestamp_end)]
            framebased_data_descriptor_frame.to_csv(framebased_data_descriptor_file_destination, index=False)
            return

        # Prepare csv files
        data_descriptor_file_path   = "data_descriptor.csv"
        data_descriptor_file_source = os.path.join(filter_dir_source, data_descriptor_file_path)

        # Extract rows between timestamp_init and timestamp_end based on frame_id column from data_descriptor.csv
        data_descriptor_frame = pd.read_csv(data_descriptor_file_source)
        data_descriptor_frame = data_descriptor_frame[
            data_descriptor_frame['timestamp_stop'].between(timestamp_init, timestamp_end)]

        frame_ids = [frame_id for frame_id in data_descriptor_frame["frame_id"]]
        framebased_data_descriptor_frame = framebased_data_descriptor_frame[
            framebased_data_descriptor_frame['frame_id'].isin(frame_ids)]
        framebased_data_descriptor_frame.to_csv(framebased_data_descriptor_file_destination, index=False)

    def generate_data_descriptor_file(self, filter_dir_source, filter_dir_destination,
                                      timestamp_init, timestamp_end, file_name):
        # Prepare csv file
        csv_file_source      = os.path.join(     filter_dir_source, file_name)
        csv_file_destination = os.path.join(filter_dir_destination, file_name)

        # Check if csv file is empty
        if self.check_csv_is_empty(csv_file_source, csv_file_destination):
            return

        # Extract rows between timestamp_init and timestamp_end
        data_frame = pd.read_csv(csv_file_source)
        data_frame = data_frame[data_frame['timestamp_stop'].between(timestamp_init, timestamp_end)]
        data_frame.to_csv(csv_file_destination, index=False)

    def create_directory(self, source_path, destination_path):
        for file_name in os.listdir(source_path):
            file_name_path = os.path.join(source_path, file_name)
            if os.path.isdir(file_name_path):
                new_source_dir      = file_name_path
                new_destination_dir = os.path.join(destination_path, file_name)
                os.mkdir(new_destination_dir)
                self.create_directory(new_source_dir, new_destination_dir)

    def create_directories_tree(self, src_name, filter_source_dir, filter_destination_dir):
        # Prepare directories
        sources_source_dir      = os.path.join(     filter_source_dir, src_name)
        sources_destination_dir = os.path.join(filter_destination_dir, src_name)

        # Create source root directory
        os.mkdir(sources_destination_dir)

        # Copy images
        self.create_directory(sources_source_dir, sources_destination_dir)

    def generate_filters(self, output_directory, filter_id, filter_type, timestamp_init, timestamp_end):
        # Create filter directory
        filter_directory_source      = os.path.join(self.input_database_dir, "datastream_{}".format(str(filter_id)))
        filter_directory_destination = os.path.join(       output_directory, "datastream_{}".format(str(filter_id)))
        os.mkdir(filter_directory_destination)

        # Prepare filter files
        for file_name in os.listdir(filter_directory_source):
            if filter_type in [5]:
                file_source = os.path.join(filter_directory_source, file_name)
                file_destination = os.path.join(filter_directory_destination, file_name)
                shutil.copyfile(file_source, file_destination)
                continue

            if file_name == "data_descriptor.csv":
                self.generate_data_descriptor_file(filter_directory_source, filter_directory_destination,
                                                   timestamp_init, timestamp_end, file_name)
            else:
                if file_name == "framebased_data_descriptor.csv":
                    self.generate_framebased_data_file(filter_directory_source, filter_directory_destination,
                                                       timestamp_init, timestamp_end, file_name)
                else:
                    if file_name == "samples":
                        self.create_directories_tree(file_name, filter_directory_source, filter_directory_destination)
                        self.copy_files(filter_directory_source, filter_directory_destination)
                    else:
                        file_source      = os.path.join(     filter_directory_source, file_name)
                        file_destination = os.path.join(filter_directory_destination, file_name)
                        shutil.copyfile(file_source, file_destination)

    def extract_subset_between_timestamps(self, timestamp_init, timestamp_end):
        print("Start extract subset from timestamp {} to {} from database {}..."
              .format(timestamp_init, timestamp_end, self.input_database_dir))

        # Prepare output directory
        output_database_dir = self.prepare_output_directory(timestamp_init, timestamp_end)

        # Prepare datablock_descriptor.csv and sampling_timestamps_sync.csv
        self.generate_main_csv_files(output_database_dir, timestamp_init, timestamp_end)

        # Prepare all filters
        datablock_descriptor_filename   = "datablock_descriptor.csv"
        datablock_descriptor_source     = os.path.join(self.input_database_dir, datablock_descriptor_filename)
        datablock_descriptor_data_frame = pd.read_csv(datablock_descriptor_source)

        for filter_id, filter_type in \
                zip(datablock_descriptor_data_frame["filter_id"], datablock_descriptor_data_frame["type"]):
            self.generate_filters(output_database_dir, filter_id, filter_type, timestamp_init, timestamp_end)

        print("Succeed extract subset from timestamp {} to {} from database {}!\nResult is found in {}!"
              .format(timestamp_init, timestamp_end, self.input_database_dir, output_database_dir))
        print("#" * 150)

    def split_equal_number_of_subsets(self, number_of_subsets):
        sampling_timestamps_sync_filename = "sampling_timestamps_sync.csv"
        sampling_timestamps_sync_source     = os.path.join(self.input_database_dir, sampling_timestamps_sync_filename)
        sampling_timestamps_sync_data_frame = pd.read_csv(sampling_timestamps_sync_source)

        number_of_samples = len(sampling_timestamps_sync_data_frame['timestamp_stop'])
        number_of_samples_for_one_subset = number_of_samples // number_of_subsets

        if number_of_subsets * number_of_samples_for_one_subset == 0:
            print("Split failed because of invalid number of subsets!")
            exit(1)

        print("Start splitting database {} in {} subsets ..."
              .format(self.input_database_dir, number_of_subsets))
        print("#" * 150)

        timestamps = sampling_timestamps_sync_data_frame['timestamp_stop']
        last_timestamp_sample = timestamps.iloc[-1]
        last_timestamp_split = None

        for timestamp_init, timestamp_end in \
                zip(timestamps[:number_of_samples-1:number_of_samples_for_one_subset],
                    timestamps[number_of_samples_for_one_subset+1:number_of_samples:number_of_samples_for_one_subset]):
            self.extract_subset_between_timestamps(int(timestamp_init), int(timestamp_end))
            last_timestamp_split = timestamp_end

        if last_timestamp_split != last_timestamp_sample:
            self.extract_subset_between_timestamps(int(last_timestamp_split), int(last_timestamp_sample))

        print("Succeed splitting database {} in {} subsets!\nResult is found in {}!"
              .format(self.input_database_dir, number_of_subsets, self.output_datasets_dir))


def main():
    spliter = SplitDatabase(input_dir=r"C:\data\Carla\carla_01",
                            output_dir=r"C:\data\Carla\carla_01_rot")
    # spliter.split_equal_number_of_subsets(number_of_subsets=5)
    spliter.extract_subset_between_timestamps(1670130361156, 1670130432192)


if __name__ == "__main__":
    main()
