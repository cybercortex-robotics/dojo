"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import os
import itertools
import csv
import numpy as np
import cv2
import struct
from .types_CyC_TYPES import CyC_FilterType, CyC_DataType

DEBUG_PARSER = False

# FOR DATA_DESCRIPTOR_CSV
TIMESTAMP_STOP_COLUMN_INDEX = 0
SAMPLING_TIME_COLUMN_INDEX = 1

class CyC_DatabaseParser(object):
    instance = None

    @staticmethod
    def get_instance(path):
        if CyC_DatabaseParser.instance is None:
            CyC_DatabaseParser.instance = CyC_DatabaseParser(path)
        return CyC_DatabaseParser.instance

    def __init__(self, database_folders):
        if DEBUG_PARSER:
            print("CyberCortex.AI Database Parser...")
            print("NOTE: Please make sure that your datablock_descriptor.csv "
                  "file matches the CyC_TYPES.h you provide")
        self.debug = True
        self.parsers = dict()

        self.key_samples = list()
        self.label_samples = list()

        if isinstance(database_folders, list):
            if len(database_folders) > 0:
                self.database_folders = list()
                if isinstance(database_folders[0], dict):
                    for dict_element in database_folders:
                        self.configure_dict(dict_element)
                else:
                    self.database_folders = database_folders
        elif isinstance(database_folders, str):
            self.database_folders = list()
            self.database_folders.append(database_folders)
        for folder in self.database_folders:
            self.parsers[folder] = create_parsers(folder)

        self.timestamp_info = dict()
        self.line_indices = dict()
        for k in self.parsers:
            for p in self.parsers[k]:
                self.timestamp_info["{}".format(p.filter_id)] = list()
                self.line_indices["{}".format(p.filter_id)] = 1

        self._calculate_timestamps()

    def close_filter_parser_by_id(self, filter_id):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    p.close_descriptor_file()

    def get_core_id(self, filter_id):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    return p.core_id

    def configure_dict(self, dict_element):
        if self.database_folders is None:
            self.database_folders = list()
        if 'path' in dict_element:
            self.database_folders.append(dict_element['path'])
        else:
            raise Exception("If you pass dict(s) for CyberCortex.AI database parser, make sure you have the path key set")
        if 'keys_samples' in dict_element:
            for filter_id in dict_element['keys_samples']:
                self.key_samples.append((dict_element['path'], filter_id))
        else:
            raise Exception("If you pass dict(s) for CyberCortex.AI database parser, make sure you have the keys_samples key set")
        if 'keys_labels' in dict_element:
            for filter_id in dict_element['keys_labels']:
                self.label_samples.append((dict_element['path'], filter_id))
        else:
            raise Exception("If you pass dict(s) for CyberCortex.AI database parser, make sure you have the keys_labels key set")

    def refresh(self, database_folders):
        if DEBUG_PARSER:
            print("CyberCortex.AI Database Parser...")
            print("NOTE: Please make sure that your datablock_descriptor.csv "
                  "file matches the CyC_TYPES.h you provide")
        self.debug = True
        self.parsers = dict()

        if isinstance(database_folders, list):
            self.database_folders = database_folders
        elif isinstance(database_folders, str):
            self.database_folders = list()
            self.database_folders.append(database_folders)
        for folder in self.database_folders:
            self.parsers[folder] = create_parsers(folder)

        self.timestamp_info = dict()
        self.line_indices = dict()
        for k in self.parsers:
            for p in self.parsers[k]:
                self.timestamp_info["{}".format(p.filter_id)] = list()
                self.line_indices["{}".format(p.filter_id)] = 1

        self._calculate_timestamps()
        CyC_DatabaseParser.instance = self

    def get_data_at_timestamp(self, filter_id, ts_stop):
        return self.read_line(filter_id, self.timestamp_to_row(filter_id, ts_stop))

    def _calculate_timestamps(self):
        with open(self.get_timesync_files()[0], "r") as ts_f:
            reader = csv.DictReader(ts_f, delimiter=',')
            for row in reader:
                ts_stop = int(row["timestamp_stop"])
                if ts_stop == -1:
                    continue
                for k, v in row.items():
                    if k is not None and int(v) != -1:
                        if k == 'timestamp_stop':
                            pass
                        else:
                            try:
                                filter_id = k.split("_")[1]
                                self.timestamp_info["{}".format(filter_id)].append(
                                    (self.line_indices["{}".format(filter_id)], int(v), ts_stop)
                                )
                                self.line_indices["{}".format(filter_id)] += 1
                            except KeyError:
                                pass
                            except IndexError:
                                pass

    def get_all_timestamps(self, filter_id):
        return self.timestamp_info["{}".format(filter_id)]

    def parse_line(self, filter_id, line_index):
        parser = None
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    parser = p
                    break
        if parser is not None:
            with open(parser.descriptor_path, "r") as desc_f:
                for i, l in enumerate(desc_f):
                    if i == line_index:
                        # Minimum number of columns: timestamp_stop, sampling_time, data
                        if len(l.strip().split(",")) < 3:
                            return None
                        return parser.parse_line(l)
        return None

    def timestamp_to_row(self, filter_id, ts_stop):
        """
        Get row from data_descriptor file which has the ts_stop specified
        :return:
        """
        line = ""
        parser = None
        for k in self.parsers:
            for p in self.parsers[k]:
                if str(p.filter_id) == str(filter_id):
                    parser = p
                    break
        if parser is not None:
            with open(parser.descriptor_path, "r") as desc_f:
                for i, l in enumerate(desc_f):
                    try:
                        if int(l.split(",")[TIMESTAMP_STOP_COLUMN_INDEX]) == ts_stop:
                            line = l
                            break
                    except ValueError:
                        pass
                    except IndexError:
                        pass
        return line

    def read_line(self, filter_id, line):
        """
        Parse line from datastream
        :param filter_id:
        :param line:
        :return:
        """
        parser = None
        data = None
        if line != "":
            for k in self.parsers:
                for p in self.parsers[k]:
                    if str(p.filter_id) == str(filter_id):
                        parser = p
                        break
            if parser is not None:
                data = parser.parse_line(line)
        return data

    def get_timesync_files(self):
        ts_files = [os.path.join(x, "sampling_timestamps_sync.csv") for x in self.database_folders]
        return ts_files

    def get_records_synced(self, filter_ids):
        """
        Return valid timestamps for synchronised data for specified filter_ids
        :param filter_ids:
        :return: a list of lists of timestamps for filter_ids
        """
        timestamps = list()
        row_names = ["datastream_{}".format(f_id) for f_id in filter_ids]
        for db_folder in self.database_folders:
            timesync_file = os.path.join(db_folder, "sampling_timestamps_sync.csv")
            with open(timesync_file, 'r') as t_f:
                reader = csv.DictReader(t_f)
                for row in reader:
                    not_synced = False
                    for k in row_names:
                        if int(row[k]) == -1:
                            not_synced = True
                            break
                    if not_synced is False:
                        row_timestamps = list()
                        for k in row_names:
                            row_timestamps.append(int(row[k]))
                        timestamps.append(row_timestamps)
        return timestamps

    def reset_generator_by_id(self, filter_id):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    p.reset()

    def reset_generator_by_type(self, filter_type):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_type == filter_type:
                    p.reset()

    def reset_all(self):
        for k in self.parsers:
            for p in self.parsers[k]:
                p.reset()

    def get_data_by_id(self, filter_id):
        valid_parsers = list()
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    valid_parsers.append(p)
        return itertools.chain.from_iterable(valid_parsers)

    def parse_and_return(self, filter_id, limit=-1):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    return p.fetch_all(limit=limit)

    def get_data_length_by_type(self, filter_type):
        num_elements = 0
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_type == filter_type:
                    num_elements += p.get_length()
        return num_elements

    def get_data_length_by_id(self, filter_id):
        num_elements = 0
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    num_elements += p.get_length()
        return num_elements

    def get_data_by_dict_samples(self):
        valid_parsers = list()
        for folder, key_sample_filter_id in self.key_samples:
            for p in self.parsers[folder]:
                if p.filter_id == key_sample_filter_id:
                    valid_parsers.append(p)
        return itertools.chain.from_iterable(valid_parsers)

    def get_data_by_dict_labels(self):
        valid_parsers = list()
        for folder, key_label_filter_id in self.label_samples:
            for p in self.parsers[folder]:
                if p.filter_id == key_label_filter_id:
                    valid_parsers.append(p)
        return itertools.chain.from_iterable(valid_parsers)

    def get_data_type_by_id(self, filter_id):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    return p.data_type

    def get_filter_type_by_id(self, filter_id):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    return p.filter_type

    def get_data_by_type(self, filter_type):
        valid_parsers = list()
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_type == filter_type:
                    valid_parsers.append(p)
        return itertools.chain.from_iterable(valid_parsers)

    def get_parser_base_path(self, filter_id):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    return p.data_base_path

    def get_filter_name_by_id(self, filter_id):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    return p.filter_name

    def get_data_shape(self, filter_id):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    return p.get_data_shape()

    def get_input_sources(self, filter_id):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    return p.input_sources

    def get_frame_id_by_timestamp(self, filter_id, ts_stop):
        for k in self.parsers:
            for p in self.parsers[k]:
                if p.filter_id == filter_id:
                    return p.get_frame_id(ts_stop)
        return -1


def create_parsers(CyC_db_base_path):
    desc_path = os.path.join(CyC_db_base_path, "datablock_descriptor.csv")
    if not os.path.exists(desc_path):
        print("DataBlock descriptor not found at path {0}. Exiting...".format(desc_path))
        exit(-1)

    parsers = list()

    with open(desc_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            core_id = int(row["vision_core_id"])  # may stay
            filter_id = int(row["filter_id"])
            filter_type = int(row["type"])
            filter_name = row["name"]
            data_type = int(row["output_data_type"])
            base_path = os.path.join(CyC_db_base_path, "datastream_{0}".format(filter_id))
            input_list = read_input_sources(row["input_sources"])
            p = create_parser(base_path=base_path,
                              core_id=core_id,
                              filter_id=filter_id,
                              filter_name=filter_name,
                              filter_type=filter_type,
                              data_type=data_type,
                              input_sources=input_list)
            if p is not None and p.is_implemented():
                parsers.append(p)

    if DEBUG_PARSER:
        print("Database located at {0} contains the following data streams:".format(CyC_db_base_path))
        idx = 1
        for p in parsers:
            print("{0}. {1}, {2}, filter_type: {3}, output_data_type: {4}".format(
                idx,
                p.filter_id,
                p.data_base_path,
                filtertype2string(p.filter_type),
                filteroutput2string(p.data_type)))
            idx += 1

    return parsers


def read_input_sources(input_src_str):
    input_list = list()
    if input_src_str:
        input_src = input_src_str.lstrip("{").rstrip("}")
        for src in input_src.split(";"):
            core_id, filter_id = src.split("-")
            input_list.append((int(core_id), int(filter_id)))
    return input_list


def filtertype2string(filter_type):
    for attr, value in vars(CyC_FilterType).items():
        if filter_type == value:
            return attr
    return "CyC_UNDEFINED_FILTER_TYPE"


def filteroutput2string(data_type):
    for attr, value in vars(CyC_DataType).items():
        if data_type == value:
            return attr
    return "CyC_UNDEFINED"


def check_prerequisites(base_path, filter_id, filter_name, filter_type, data_type, input_sources):
    if not os.path.exists(base_path):
        print("Warning: Base path {} does not exist".format(base_path))
        return False

    desc_path = os.path.join(base_path, "data_descriptor.csv")
    if not os.path.exists(desc_path):
        print("File {} does not exist".format(desc_path))
        return False

    if data_type == CyC_DataType.CyC_IMAGE:
        samples_dir = os.path.join(base_path, "samples")

        if not os.path.exists(samples_dir):
            print("Samples directory {} does not exist".format(samples_dir))
            return False

        found_left = False
        found_right = False
        for root, dirs, files in os.walk(samples_dir):
            for dir in dirs:
                if "left" in dir:
                    found_left = True
                if "right" in dir:
                    found_right = True

        # if not os.path.exists(os.path.join(samples_dir, "left")):
        #     print("Left image directory {} does not exist".format(os.path.join(samples_dir, "left")))
        #     return False

        if filter_type == CyC_FilterType.CyC_RGBDCAMERA_FILTER_TYPE:
            if not found_left or not found_right:
                print("Right image directory or left image directory does not exist")
                return False
        else:
            if not found_left:
                pass

    if data_type == CyC_DataType.CyC_2D_ROIS or data_type == CyC_DataType.CyC_REFERENCE_SETPOINTS:
        framebase_desc_path = os.path.join(base_path, "framebased_data_descriptor.csv")
        if not os.path.exists(framebase_desc_path):
            print("Framebased descriptor path {} does not exist".format(framebase_desc_path))
            return False

    return True


def create_parser(base_path, core_id, filter_id, filter_name, filter_type, data_type, input_sources):
    if not check_prerequisites(base_path, filter_id, filter_name, filter_type, data_type, input_sources):
        print("Warning: Parser for filter {} can not be constructed".format(filter_id))
        return None

    if data_type == CyC_DataType.CyC_VECTOR_INT:
        return VectorIntReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_VECTOR_FLOAT:
        return VectorFloatReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_VECTOR_FLOAT:
        return VectorFloatReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_VECTOR_DOUBLE:
        return VectorDoubleReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_IMAGE:
        return ImageReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_ULTRASONICS:
        return UltrasonicsReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_IMU:
        return ImuReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_GPS:
        return GpsReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_RADAR:
        return RadarReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_POINTS and \
            filter_type != CyC_FilterType.CyC_VEHICLE_MISSION_PLANNER_FILTER_TYPE:
        # TODO: Verify in mission planner filter is / will be supported
        return PointsReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_VOXELS:
        return VoxelsReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_POSES_6D:
        return Pose6DReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_2D_ROIS:
        return Rois2DReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_3D_BBOXES:
        return Rois3DBboxesReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_OCTREE:
        return OctreeReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_GRIDMAP:
        return GridmapReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_REFERENCE_SETPOINTS:
        return ReferenceSetpointsReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_STATE:
        return StateReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_CONTROL_INPUT:
        return ControlInputReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    # elif data_type == CyC_DataType.CyC_MEASUREMENT:
    #    return MeasurementReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_ESTIMATED_TRAJECTORY:
        return EstimatedTrajectoryReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    elif data_type == CyC_DataType.CyC_LANES_MODEL:
        return LaneModelReader(base_path, core_id, filter_id, filter_name, filter_type, input_sources)
    else:
        print("WARNING: Filter with output type {0} not implemented".format(filteroutput2string(data_type)))
        return None


class DataTypeFileReader(object):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, data_type, input_sources):
        self.data_base_path = data_base_path
        self.core_id = core_id
        self.filter_id = filter_id
        self.filter_name = filter_name
        self.filter_type = filter_type
        self.data_type = data_type
        self.input_sources = input_sources
        self.descriptor_path = os.path.join(data_base_path, "data_descriptor.csv")

        if not os.path.exists(self.descriptor_path):
            print("Descriptor file {0} does not exist. Exiting.".format(self.descriptor_path))
            exit(-1)

        with open(self.descriptor_path, "r") as f:
            self.length = sum(1 for _ in f) - 1  # don't count the header

        self.fp = open(self.descriptor_path, "r")
        self.header = self.fp.readline()  # header
        self._is_implemented = False
        self.data_shape = None

    def close_descriptor_file(self):
        self.fp.close()

    def get_data_shape(self):
        return self.data_shape

    def reset(self):
        self.fp.close()
        self.fp = open(self.descriptor_path, "r")
        self.header = self.fp.readline()  # header

    def __iter__(self):
        return self

    def is_implemented(self):
        return self._is_implemented

    def get_length(self):
        return self.length

    def parse_line(self, line):
        raise NotImplementedError

    def get_frame_id(self, ts_stop):
        return -1

    def fetch_all(self, limit):
        raise NotImplementedError

    def __next__(self):
        try:
            line = self.fp.readline()
        except ValueError:
            self.reset()
            line = self.fp.readline()

        if len(line):
            try:
                return self.parse_line(line)
            except IndexError:
                print("Malformed record in file {0}, line {1}. Exiting...".format(self.descriptor_path, line))
                exit(-2)
        else:
            self.fp.close()
            raise StopIteration


class VectorIntReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(VectorIntReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                              CyC_DataType.CyC_VECTOR_INT, input_sources)
        self._is_implemented = True

    def parse_line(self, line):
        """
        :param line: timestamp_stop, sampling_time, ...
        :return: timestamp_stop, sampling_time, [data]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])
        data = [int(x) for x in s_line[2:]]
        return timestamp_stop, sampling_time, data


class VectorFloatReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(VectorFloatReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                                CyC_DataType.CyC_VECTOR_FLOAT, input_sources)

    def parse_line(self, line):
        raise NotImplementedError


class VectorDoubleReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(VectorDoubleReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                                 CyC_DataType.CyC_VECTOR_DOUBLE, input_sources)

    def parse_line(self, line):
        raise NotImplementedError


class ImageReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(ImageReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                          CyC_DataType.CyC_IMAGE, input_sources)
        self._check_image_filter_type()

        self.frame_desc_file = os.path.join(self.data_base_path, "framebased_data_descriptor.csv")
        if self.filter_type == CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE and \
                not os.path.exists(self.frame_desc_file):
            print("framebased_data_descriptor.csv not found at path {0}".format(self.frame_desc_file))

        if self.filter_type == CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE:
            if os.path.exists(self.frame_desc_file):
                with open(self.frame_desc_file, "r") as desc_f:
                    lines = desc_f.readlines()
                header = lines[0]
                header_items = header.strip().split(",")
                found_instance = False
                for it in header_items:
                    if it == "instance":
                        found_instance = True
                        break
                if not found_instance:
                    header_items.insert(3, "instance")
                    for idx in range(1, len(lines)):
                        l_s = lines[idx].strip().split(",")
                        l_s.insert(3, "0")
                        new_line = ",".join([l for l in l_s])
                        lines[idx] = new_line
                    lines[0] = ",".join(header_items)
                    with open(self.frame_desc_file, "w") as desc_f:
                        for line in lines:
                            desc_f.write(line + "\n")

        self._is_implemented = True

    def _check_image_filter_type(self):
        if self.filter_type not in [CyC_FilterType.CyC_MONO_CAMERA_FILTER_TYPE,
                                    CyC_FilterType.CyC_RGBDCAMERA_FILTER_TYPE,
                                    CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE]:
            print("WARNING: Filter type {0} not supported yet. Parser could fail".format(
                filtertype2string(self.filter_type))
            )

    def parse_line(self, line):
        """
        :param line: timestamp_stop, sampling_time,
                     left_image_path/left_file_path/semantic, right_image_path/right_file_path/instances
        :return: timestamp_stop, sampling_time, [us_sequences]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])

        left_image = None
        right_image = None
        left_img_idx = None
        right_img_idx = None

        for i, k in enumerate(self.header.split(",")):
            if "left_image_path" in k or "left_file_path" in k or "semantic" in k:
                left_img_idx = i
            elif "right_image_path" in k or "right_file_path" in k or "instances" in k:
                right_img_idx = i

        try:
            if left_img_idx:
                left_image_relative_path = s_line[left_img_idx].strip()
                if left_image_relative_path != str(-1) and left_image_relative_path != "":
                    left_image_path = os.path.join(self.data_base_path, left_image_relative_path)
                    if self.filter_type == CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE:
                        left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        left_image = cv2.imread(left_image_path)
                    if left_image is not None and left_image.shape[-1] == 3:
                        left_image = left_image[..., ::-1].copy()

            if right_img_idx:
                right_image_relative_path = s_line[right_img_idx].strip()
                if right_image_relative_path != str(-1) and right_image_relative_path != "":
                    right_image_path = os.path.join(self.data_base_path, right_image_relative_path)
                    if self.filter_type == CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE:
                        right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        right_image = cv2.imread(right_image_path)
                    if right_image is not None and right_image.shape[-1] == 3:
                        right_image = right_image[..., ::-1].copy()

            if left_image is not None:
                self.data_shape = left_image.shape[:2]

            if self.filter_type == CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE:
                shapes = self.parse_shapes_line(line)
                return timestamp_stop, sampling_time, left_image, right_image, shapes
        except IndexError:
            if self.filter_type == CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE:
                return None, None, None, None, None, None
            return None, None, None, None, None
        except ValueError:
            if self.filter_type == CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE:
                return None, None, None, None, None, None
            return None, None, None, None, None

        timestamp_image = int(s_line[SAMPLING_TIME_COLUMN_INDEX + 1])
        return timestamp_stop, sampling_time, timestamp_image, left_image, right_image

    def parse_shapes_line(self, line):
        """
        :param line: timestamp_stop, sampling_time, semantic, instances
        :return: [shapes]
        """
        try:
            s_line = line.split(",")
            timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
            shapes = list()
        except ValueError:
            return None
        except IndexError:
            return None

        if not os.path.exists(self.frame_desc_file):
            return None

        # timestamp_stop, shape_id, cls, instance, points
        with open(self.frame_desc_file, "r") as f:
            trigger = False
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["timestamp_stop"]) == timestamp_stop:
                    trigger = True
                    shape = list()
                    shape.append(float(row["timestamp_stop"]))
                    shape.append(float(row["shape_id"]))
                    shape.append(float(row["cls"]))

                    if "instance" in row:
                        shape.append(int(row["instance"]))
                    else:
                        shape.append(0)

                    try:
                        points_str = row["points"][2:-2]
                        points_str = points_str.split("][")
                        points = list()

                        for point in points_str:
                            point = point.split(" ")
                            points.append([float(point[0]), float(point[1])])

                        shape.append(points)
                        shapes.append(shape)
                    except:
                        continue
                else:
                    if trigger:
                        break
        return shapes


class UltrasonicsReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(UltrasonicsReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                                CyC_DataType.CyC_ULTRASONICS, input_sources)
        self._is_implemented = True

    def _get_us_config(self):
        idx1 = 0
        idx2 = 0
        header = self.header.split(",")
        for k in header:
            if "sonic" in k:
                try:
                    idx1 = int(k.split("_")[1])
                    idx2 = int(k.split("_")[2])
                except IndexError:
                    if len(k.split("_")) == 2:
                        idx1 = 0
                        idx2 = int(k.split("_")[1])
                    else:
                        print("Error: Could not parse Ultrasonic file. Exiting...")
                        exit(-3)
        return idx1 + 1, idx2 + 1

    def parse_line(self, line):
        """
        :param line: timestamp_stop,sampling_time,sonic_0, ..., sonic_33
        :return: timestamp_stop, sampling_time, [us_sequences]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])

        num_sequences, size_us = self._get_us_config()
        us_sequences = list()
        s_line = line.split(",")
        us_line = s_line[SAMPLING_TIME_COLUMN_INDEX+1:]
        for idx0 in range(num_sequences):
            us_sensor_array = list()
            for idx1 in range(size_us):
                us_sensor_array.append(float(us_line[idx0 * size_us + idx1]))
            us_sequences.append(us_sensor_array)
        if len(us_sequences) == 1:  # if we don't have multiple sequences, return just one unnested sequence
            us_sequences = us_sequences[0]

        return timestamp_stop, sampling_time, np.array(us_sequences, dtype=np.float32)


class ImuReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(ImuReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                        CyC_DataType.CyC_IMU, input_sources)
        self._is_implemented = True

    def parse_line(self, line):
        """
         :param line: timestamp_stop,sampling_time,imu_file_path
         :return: timestamp_stop, sampling_time, [values]
         """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])
        values = []

        fp = os.path.join(self.data_base_path, s_line[SAMPLING_TIME_COLUMN_INDEX + 1].strip("\n"))
        with open(fp, "r") as lidar_data:
            # timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
            for line in lidar_data:
                try:
                    value = []
                    line = line.strip("\n").split(",")
                    data = [float(x) for x in line[1:]]
                    value.append(data)
                    value.append(line[0])
                    values.append(value)
                except ValueError:
                    pass
                except IndexError:
                    pass

        return timestamp_stop, sampling_time, values


class GpsReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(GpsReader, self).__init__(data_base_path, core_id, filter_id, filter_name,
                                        filter_type, CyC_DataType.CyC_GPS, input_sources)
        self._is_implemented = True

    def parse_line(self, line):
        """
        :param line: timestamp_stop, sampling_time, lat, lng, alt
        :return: timestamp_stop, sampling_time, [data]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])
        data = [float(x) for x in s_line[2:]]
        return timestamp_stop, sampling_time, np.array(data, dtype=np.float32)


class RadarReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(RadarReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                          CyC_DataType.CyC_RADAR, input_sources)
        self._is_implemented = True

    def parse_line(self, line):
        """
        :param line: timestamp_stop, sampling_time, radar_file_path_0
        :return: timestamp_stop, sampling_time, [values]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])

        fp = os.path.join(self.data_base_path, s_line[SAMPLING_TIME_COLUMN_INDEX + 1].strip("\n"))
        with open(fp, "r") as lidar_data:
            for line in lidar_data:
                try:
                    elements = list()
                    for x in line.split(" "):
                        elements.append(float(x))
                    values.append(elements)
                except ValueError:
                    pass
                except IndexError:
                    pass

        return timestamp_stop, sampling_time, np.array(values, dtype=np.float32)


class PointsReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(PointsReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                           CyC_DataType.CyC_POINTS, input_sources)
        self.frame_desc_file = os.path.join(self.data_base_path, "framebased_data_descriptor.csv")
        if not os.path.exists(self.frame_desc_file):
            print("framebased_data_descriptor.csv not found at path {0}".format(self.frame_desc_file))
            exit(-1)
        self._is_implemented = True

    # TODO Check
    def get_frame_id(self, ts_stop):
        lines = self.fp.readlines()
        ret_val = -1
        for line in lines:
            stop = int(line.split(",")[0])
            if ts_stop == stop:
                ret_val = int(line.strip().split(",")[-1])
        self.reset()
        return ret_val

    def parse_line(self, line):
        """
        :param line: timestamp_stop,sampling_time,lidar_file_path_0
        :return: timestamp_stop, sampling_time, [points]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])
        frame_id = int(s_line[SAMPLING_TIME_COLUMN_INDEX+1])

        points = list()
        # frame_id, x, y, id, score
        with open(self.frame_desc_file, "r") as f:
            trigger = False
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["frame_id"]) == frame_id:
                    trigger = True
                    point = list()
                    point.append(float(row["x"]))
                    point.append(float(row["y"]))
                    point.append(float(row["id"]))
                    point.append(float(row["score"]))
                    points.append(np.array(point, dtype=np.float32))
                else:
                    if trigger:
                        break

        return timestamp_stop, sampling_time, points


class VoxelsReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(VoxelsReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                           CyC_DataType.CyC_VOXELS, input_sources)
        if not self.filter_type == CyC_FilterType.CyC_LIDAR_FILTER_TYPE:
            print("Voxels reader not yet implemented for filter type different than: ",
                  filtertype2string(CyC_FilterType.CyC_LIDAR_FILTER_TYPE))
            exit(12)
        self._is_implemented = True

    # TODO Check
    def _parse_lidar_binary(self, lidar_file):
        lidar_pts = list()
        with open(lidar_file, "rb") as lidar_f:
            num_pts = int(struct.unpack("q", lidar_f.read(8))[0])
            for idx in range(num_pts):
                id, score, x, y, z, w = struct.unpack("ifffff", lidar_f.read(24))
                pt = list()
                pt.append(float(x))
                pt.append(float(y))
                pt.append(float(z))
                lidar_pts.append(pt)
        lidar_pts = np.array(lidar_pts, dtype=np.float32)
        return lidar_pts

    def parse_line(self, line):
        """
        :param line: timestamp_stop,sampling_time,lidar_file_path_0
        :return: timestamp_stop, sampling_time, [values]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])

        fp = os.path.join(self.data_base_path, s_line[SAMPLING_TIME_COLUMN_INDEX+1].strip())
        if fp.endswith(".ply"):
            values = list()
            with open(fp, "r") as lidar_data:
                for line in lidar_data:
                    try:
                        s_data = line.split(" ")
                        x = float(s_data[0])
                        y = float(s_data[1])
                        z = float(s_data[2])
                        values.append([x, y, z])
                    except ValueError:
                        pass
                    except IndexError:
                        pass
            values = np.array(values, dtype=np.float32)
        elif fp.endswith(".data"):
            values = self._parse_lidar_binary(fp)
        else:
            print("Lidar file format not supported. Exiting...")
            values = None
            exit(1)

        return timestamp_stop, sampling_time, values


class Pose6DReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(Pose6DReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                           CyC_DataType.CyC_POSE_6D, input_sources)

    def parse_line(self, line):
        raise NotImplementedError


class Rois2DReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(Rois2DReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                           CyC_DataType.CyC_2D_ROIS, input_sources)

        self.frame_desc_file = os.path.join(self.data_base_path, "framebased_data_descriptor.csv")
        if not os.path.exists(self.frame_desc_file):
            print("framebased_data_descriptor.csv not found at path {0}".format(self.frame_desc_file))
            exit(-1)

        self._rois_dict = dict()
        self._is_implemented = True

    def get_frame_id(self, ts_stop):
        lines = self.fp.readlines()
        ret_val = -1
        for line in lines:
            stop = int(line.split(",")[0])
            if ts_stop == stop:
                ret_val = int(line.strip().split(",")[-1])
        self.reset()
        return ret_val

    def fetch_all(self, limit):
        if self.fp.closed is False:
            self.fp.close()
        self.fp = open(self.descriptor_path, "r")
        self.fp.readline()  # Skip header
        lines = self.fp.readlines()
        ret_vec = list()
        idx = 0
        with open(self.frame_desc_file) as f:
            lines_framebased = f.readlines()
        self._collate_rois(lines_framebased)
        for line in lines:
            if limit != -1 and idx >= limit:
                break
            idx += 1
            ret_vec.append(self._parse_line(line, lines_framebased))
            # ret_vec.append(self.parse_line(line))

        self.fp.close()

        # Reopen file
        self.fp = open(self.descriptor_path, "r")
        self.fp.readline()  # Skip header

        return ret_vec

    def _collate_rois(self, lines_framebased):
        for s in lines_framebased:
            row = s.split(",")
            try:
                frame_id = int(row[0])
                if frame_id in self._rois_dict:
                    self._rois_dict[frame_id].append(np.array([float(r) for r in row]))
                else:
                    self._rois_dict[frame_id] = [np.array([float(r) for r in row])]
            except ValueError:
                continue

    def _parse_line(self, line, lines_framebased):
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])
        frame_id = int(s_line[SAMPLING_TIME_COLUMN_INDEX+1])

        try:
            rois = self._rois_dict[frame_id]
        except KeyError:
            rois = []
            # Uncomment for debug purposes
            # print("Warning: no rois found for frame id ", frame_id)

        return timestamp_stop, sampling_time, frame_id, rois

    def parse_line(self, line):
        """
        :param line: timestamp_stop, sampling_time, frame_id
        :return: timestamp_stop, sampling_time, [rois]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])
        frame_id = int(s_line[SAMPLING_TIME_COLUMN_INDEX+1])

        rois = list()
        # frame_id, roi_id, cls, x, y, width, height
        with open(self.frame_desc_file, "r") as f:
            trigger = False
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["frame_id"]) == frame_id:
                    trigger = True
                    roi = list()
                    roi.append(float(row["frame_id"]))
                    roi.append(float(row["roi_id"]))
                    roi.append(float(row["cls"]))
                    roi.append(float(row["x"]))
                    roi.append(float(row["y"]))
                    roi.append(float(row["width"]))
                    roi.append(float(row["height"]))
                    rois.append(np.array(roi, dtype=np.float32))
                else:
                    if trigger:
                        break

        return timestamp_stop, sampling_time, rois


class Rois2DReader_ROAD(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(Rois2DReader_ROAD, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                                CyC_DataType.CyC_2D_ROIS, input_sources)
        self.frame_desc_file = os.path.join(self.data_base_path, "framebased_data_descriptor.csv")
        if not os.path.exists(self.frame_desc_file):
            print("framebased_data_descriptor.csv not found at path {0}".format(self.frame_desc_file))
            exit(-1)
        self._is_implemented = True

    # TODO Check
    def fetch_all(self, limit):
        if self.fp.closed is False:
            self.fp.close()
        self.fp = open(self.descriptor_path, "r")
        self.fp.readline()  # Skip header
        lines = self.fp.readlines()
        ret_vec = list()
        idx = 0
        with open(self.frame_desc_file) as f:
            lines_framebased = f.readlines()
        for line in lines:
            if limit != -1 and idx >= limit:
                break
            if idx % 500 == 0:
                print("Processed {}:{}".format(
                    idx + 1, self.length
                ))
            idx += 1
            ret_vec.append(self._parse_line(line, lines_framebased))
            # ret_vec.append(self.parse_line(line))

        self.fp.close()

        # Reopen file
        self.fp = open(self.descriptor_path, "r")
        self.fp.readline()  # Skip header

        return ret_vec

    # TODO Check
    def _parse_line(self, line, lines_framebased):
        s_line = line.split(",")
        frame_id = int(s_line[3])
        rois = list()
        with open(self.frame_desc_file, "r") as f:
            trigger = False
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["frame_id"]) == frame_id:
                    trigger = True
                    roi = list()
                    roi.append(float(row["frame_id"]))
                    roi.append(float(row["roi_id"]))
                    roi.append(float(row["cls"]))
                    roi.append(float(row["x"]))
                    roi.append(float(row["y"]))
                    roi.append(float(row["width"]))
                    roi.append(float(row["height"]))

                    actions_string = row["action"].strip("[]").split(";")
                    actions = list()
                    for action in actions_string:
                        actions.append(float(action))

                    locs_string = row["loc"].strip("[]").split(";")
                    locs = list()
                    for loc in locs_string:
                        locs.append(float(loc))
                    roi.append(actions)
                    roi.append(locs)
                    rois.append(np.array(roi))
                else:
                    if trigger:
                        break
        return int(s_line[0]), int(s_line[1]), int(s_line[2]), int(s_line[4]), rois

    def parse_line(self, line):
        """
        :param line: timestamp_stop, sampling_time, frame_id, ego_id
        :return: timestamp_stop, sampling_time, [rois]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])
        frame_id = int(s_line[SAMPLING_TIME_COLUMN_INDEX+1])
        ego_id = int(s_line[SAMPLING_TIME_COLUMN_INDEX + 2])

        rois = list()
        # frame_id, roi_id, cls, x, y, width, height, action, loc
        with open(self.frame_desc_file, "r") as f:
            trigger = False
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["frame_id"]) == frame_id:
                    trigger = True
                    roi = list()
                    roi.append(float(row["frame_id"]))
                    roi.append(float(row["roi_id"]))
                    roi.append(float(row["cls"]))
                    roi.append(float(row["x"]))
                    roi.append(float(row["y"]))
                    roi.append(float(row["width"]))
                    roi.append(float(row["height"]))

                    actions_string = row["action"].strip("[]").split(";")
                    actions = list()
                    for action in actions_string:
                        actions.append(float(action))

                    locs_string = row["loc"].strip("[]").split(";")
                    locs = list()
                    for loc in locs_string:
                        locs.append(float(loc))
                    roi.append(actions)
                    roi.append(locs)
                    rois.append(np.array(roi))
                else:
                    if trigger:
                        break

        # TODO Check
        return timestamp_stop, sampling_time, ego_id, rois


class Rois3DBboxesReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(Rois3DBboxesReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                                 CyC_DataType.CyC_3D_BBOXES, input_sources)
        self.frame_desc_file = os.path.join(self.data_base_path, "framebased_data_descriptor.csv")
        if not os.path.exists(self.frame_desc_file):
            print("framebased_data_descriptor.csv not found at path {0}".format(self.frame_desc_file))
            exit(-1)
        self._is_implemented = True

    def parse_line(self, line):
        """
        :param line: timestamp_stop, sampling_time, frame_id
        :return: timestamp_stop, sampling_time, [rois]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])
        frame_id = int(s_line[SAMPLING_TIME_COLUMN_INDEX+1])

        rois = list()
        try:
            # frame_id, roi_id, cls, x, y, z, w, h, l, roll, pitch, yaw
            with open(self.frame_desc_file, "r") as f:
                trigger = False
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row["frame_id"]) == frame_id:
                        trigger = True
                        roi = list()
                        roi.append(float(row["frame_id"]))
                        roi.append(float(row["roi_id"]))
                        roi.append(float(row["cls"]))
                        roi.append(float(row["x"]))
                        roi.append(float(row["y"]))
                        roi.append(float(row["z"]))
                        roi.append(float(row["w"]))
                        roi.append(float(row["h"]))
                        roi.append(float(row["l"]))
                        roi.append(float(row["roll"]))
                        roi.append(float(row["pitch"]))
                        roi.append(float(row["yaw"]))
                        rois.append(np.array(roi))
                    else:
                        if trigger:
                            break
            return timestamp_stop, sampling_time, rois
        except IndexError:
            return None
        except ValueError:
            return None

    # TODO Check
    def fetch_all(self, limit):
        if self.fp.closed is False:
            self.fp.close()

        self.fp = open(self.descriptor_path, "r")
        self.fp.readline()  # Skip header
        lines = self.fp.readlines()

        ret_vec = list()
        idx = 0
        with open(self.frame_desc_file) as f:
            lines_framebased = f.readlines()

        for line in lines:
            if limit != -1 and idx >= limit:
                break
            idx += 1
            ret_vec.append(self.parse_line(line))

        self.fp.close()

        # Reopen file
        self.fp = open(self.descriptor_path, "r")
        self.fp.readline()  # Skip header

        return ret_vec


class OctreeReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(OctreeReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                           CyC_DataType.CyC_OCTREE, input_sources)

    def parse_line(self, line):
        raise NotImplementedError


class GridmapReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(GridmapReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                            CyC_DataType.CyC_GRIDMAP, input_sources)

    def parse_line(self, line):
        raise NotImplementedError


class ReferenceSetpointsReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(ReferenceSetpointsReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                                       CyC_DataType.CyC_REFERENCE_SETPOINTS, input_sources)
        self.frame_desc_file = os.path.join(self.data_base_path, "framebased_data_descriptor.csv")
        if not os.path.exists(self.frame_desc_file):
            print("framebased_data_descriptor.csv not found at path {0}".format(self.frame_desc_file))
            exit(-1)
        self._is_implemented = True

    def parse_line(self, line):
        """
        :param line: timestamp_stop,sampling_time,frame_id ?
        :return: timestamp_stop, sampling_time, [reference_setpoints]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])
        frame_id = int(s_line[SAMPLING_TIME_COLUMN_INDEX+1])

        reference_setpoints = list()

        # frame_id, ref_point_0, ref_point_2, ref_point_3
        with open(self.frame_desc_file, "r") as f:
            trigger = False
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["frame_id"]) == frame_id:
                    trigger = True
                    reference_setpoints.append(np.array([float(row["ref_point_0"]),
                                                         float(row["ref_point_1"]),
                                                         float(row["ref_point_2"]),
                                                         float(row["ref_point_3"])], dtype=np.float32))
                else:
                    if trigger:
                        break

        return timestamp_stop, sampling_time, reference_setpoints


class StateReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(StateReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                          CyC_DataType.CyC_STATE, input_sources)
        self._is_implemented = True

    def parse_line(self, line):
        """
        :param line: timestamp_stop,sampling_time,state_variable_0,state_variable_1,state_variable_2,state_variable_3
        :return: timestamp_stop, sampling_time, [data]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])
        data = [float(x) for x in s_line[SAMPLING_TIME_COLUMN_INDEX+1:SAMPLING_TIME_COLUMN_INDEX+5]]
        return timestamp_stop, sampling_time, np.array(data, dtype=np.float32)


class ControlInputReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(ControlInputReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                                 CyC_DataType.CyC_CONTROL_INPUT, input_sources)
        self._is_implemented = True

    def parse_line(self, line):
        """
        :param line: timestamp_stop, sampling_time, ?, ?
        :return: timestamp_stop, sampling_time, [?, ?]
        """
        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])
        return timestamp_stop, sampling_time, np.array([float(s_line[2]), float(s_line[3])], dtype=np.float32)


class EstimatedTrajectoryReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(EstimatedTrajectoryReader, self).__init__(
            data_base_path, core_id, filter_id, filter_name,
            filter_type, CyC_DataType.CyC_ESTIMATED_TRAJECTORY, input_sources
        )

    def parse_line(self, line):
        raise NotImplementedError


class SegmentationReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(SegmentationReader, self).__init__(data_base_path, core_id, filter_id, filter_name,
                                                 filter_type, CyC_DataType.CyC_IMAGE, input_sources)

        self.frame_desc_file = os.path.join(self.data_base_path, "framebased_data_descriptor.csv")
        if not os.path.exists(self.frame_desc_file):
            print("framebased_data_descriptor.csv not found at path {0}".format(self.frame_desc_file))

        self._is_implemented = True

    def parse_line(self, line):
        """
        :param line: timestamp_stop,sampling_time,semantic,instances
        :return: timestamp_stop, sampling_time, semseg_img, instance_img, additional_image, [shapes]
        """
        semseg_img, instance_img, additional_img = None, None, None
        semseg_img_idx, instance_img_idx, additional_img_idx = None, None, None
        shapes = list()

        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])

        for i, k in enumerate(self.header.split(",")):
            if "semseg_img" in k:
                semseg_img_idx = i
            elif "instance_img" in k:
                right_img_idx = i
            elif "annot_img" in k:
                additional_img_idx = i

        if semseg_img_idx:
            semseg_img_relative_path = s_line[semseg_img_idx].strip()
            if semseg_img_relative_path != str(-1) and semseg_img_relative_path != "":
                semseg_img_path = os.path.join(self.data_base_path, semseg_img_relative_path)
                semseg_img = cv2.imread(semseg_img_path)
                if semseg_img is not None:
                    semseg_img = semseg_img[..., ::-1].copy()

        if instance_img_idx:
            instance_img_relative_path = s_line[instance_img_idx].strip()
            if instance_img_relative_path != str(-1) and instance_img_relative_path != "":
                instance_img_path = os.path.join(self.data_base_path, instance_img_relative_path)
                instance_img = cv2.imread(instance_img_path)
                if instance_img is not None:
                    instance_img = instance_img[..., ::-1].copy()

        if semseg_img is not None:
            self.data_shape = semseg_img.shape[:2]

        if additional_img_idx:
            additional_image_relative_path = s_line[additional_img_idx].strip()
            if additional_image_relative_path != str(-1) and additional_image_relative_path != "":
                additional_image_path = os.path.join(self.data_base_path, additional_image_relative_path)
                additional_image = cv2.imread(additional_image_path)
                if additional_image is not None:
                    additional_image = additional_image[..., ::-1].copy()

        # timestamp_stop, shape_id, cls, instance, points
        with open(self.frame_desc_file, "r") as f:
            trigger = False
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["timestamp_stop"]) == timestamp_stop:
                    trigger = True
                    shape = list()
                    shape.append(float(row["timestamp_stop"]))
                    shape.append(float(row["shape_id"]))
                    shape.append(float(row["cls"]))

                    points_str = row["points"][2:-2]
                    points_str = points_str.split("][")
                    points = list()

                    for point in points_str:
                        point = point.split(" ")
                        points.append([float(point[0]), float(point[1])])

                    shape.append(points)
                    shapes.append(shape)
                else:
                    if trigger:
                        break

        return timestamp_stop, sampling_time, semseg_img, instance_img, additional_image, shapes


class LaneModelReader(DataTypeFileReader):
    def __init__(self, data_base_path, core_id, filter_id, filter_name, filter_type, input_sources):
        super(LaneModelReader, self).__init__(data_base_path, core_id, filter_id, filter_name, filter_type,
                                              CyC_DataType.CyC_LANES_MODEL, input_sources)
        self.frame_desc_file = os.path.join(self.data_base_path, "framebased_data_descriptor.csv")
        if not os.path.exists(self.frame_desc_file):
            print("framebased_data_descriptor.csv not found at path {0}".format(self.frame_desc_file))
            exit(-1)
        self._is_implemented = True

    def parse_line(self, line):
        """
        :param line: timestamp_stop, sampling_time, semantic, instances
        :return: timestamp_stop, sampling_time, semantic_image, instances_image, {lanes}, {thetas}
        """
        seg_img_idx = None
        inst_img_idx = None
        seg_img = None
        inst_img = None
        lanes = {}
        thetas = {}

        s_line = line.split(",")
        timestamp_stop = int(s_line[TIMESTAMP_STOP_COLUMN_INDEX])
        sampling_time = int(s_line[SAMPLING_TIME_COLUMN_INDEX])

        for i, k in enumerate(self.header.split(",")):
            if "left_image_path" in k or "left_file_path" in k or "semantic" in k:
                seg_img_idx = i
            elif "right_image_path" in k or "right_file_path" in k or "instances" in k:
                inst_img_idx = i

        if seg_img_idx is not None:
            seg_img_relative_path = s_line[seg_img_idx].strip()
            if seg_img_relative_path != str(-1) and seg_img_relative_path != "":
                seg_img_path = os.path.join(self.data_base_path, seg_img_relative_path)
                seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)

        if inst_img_idx is not None:
            inst_img_relative_path = s_line[inst_img_idx].strip()
            if inst_img_relative_path != str(-1) and seg_img_relative_path != "":
                inst_img_path = os.path.join(self.data_base_path, inst_img_relative_path)
                inst_img = cv2.imread(inst_img_path, cv2.IMREAD_GRAYSCALE)

        # timestamp_stop,lane_id,points,theta_0,theta_1,theta_2,theta_3
        with open(self.frame_desc_file, "r") as f:
            trigger = False
            reader = csv.DictReader(f)

            for row in reader:
                if int(row["timestamp_stop"]) == timestamp_stop:
                    trigger = True

                    # Lanes
                    lane = []
                    if row['points'] is not None and row['points'] != '[]':
                        lane_id = int(row["lane_id"])
                        lane_str = row["points"]
                        lane_str = lane_str.strip("[").strip(']')
                        lane = [int(val) for val in lane_str.split(" ")]
                        lanes[lane_id] = lane

                    # Thetas
                    if row['theta_0'] is not None and row['theta_0'] != '':
                        theta = [float(row['theta_0']), float(row['theta_1']),
                                 float(row['theta_2']), float(row['theta_3'])]
                        thetas[lane_id] = theta
                else:
                    if trigger:
                        break

        return timestamp_stop, sampling_time, seg_img, inst_img, lanes, thetas
