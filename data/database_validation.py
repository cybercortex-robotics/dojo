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
from argparse import Namespace
from PyQt5.QtCore import QObject, pyqtSignal

import json
import libconf
import io


class LogSignals(QObject):
    info_signal = pyqtSignal(int, str)
    warning_signal = pyqtSignal(int, str)
    error_signal = pyqtSignal(int, str)


class ReturnCodes:
    UNKNOWN_ERROR = -1
    OK = 0
    CONFIG_NOT_FOUND = 1
    INVALID_CONFIG_JSON = 2
    INVALID_CALIBRATION_FILE = 3
    BLOCKCHAIN_NOT_FOUND = 4
    TIMESTAMP_SYNC_NOT_FOUND = 5
    DATASTREAM_FOLDER_NOT_FOUND = 6
    BLOCKCHAIN_FILE_HEADER_WRONG = 7
    FILTER_TYPE_INVALID = 8
    DATA_TYPE_INVALID = 9
    CORE_ID_INVALID = 10
    FILTER_ID_INVALID = 11
    INVALID_BLOCKCHAIN_LINE = 12
    OBJECT_CLASSES_NOT_FOUND = 13
    OBJECT_CLASSES_INVALID = 14
    INVALID_CALIBRATION_FILE_EXTENSION = 15
    DB_PATH_NOT_DIR = 16
    CALIBRATION_FILE_NOT_FILE = 17
    CALIBRATION_FILE_NOT_FOUND = 18
    DB_PATH_NOT_FOUND = 19
    OBJECT_CLASSES_NOT_FILE = 20
    INVALID_OBJECT_CLASSES_FILE_EXTENSION = 21

    @staticmethod
    def code2string(code):
        if code == ReturnCodes.OK:
            return "OK"
        elif code == ReturnCodes.CONFIG_NOT_FOUND:
            return "Config file not found"
        elif code == ReturnCodes.INVALID_CONFIG_JSON:
            return "Invalid configuration json"
        elif code == ReturnCodes.INVALID_CALIBRATION_FILE:
            return "Invalid camera calibration file"
        elif code == ReturnCodes.BLOCKCHAIN_NOT_FOUND:
            return "Blockchain file not found"
        elif code == ReturnCodes.TIMESTAMP_SYNC_NOT_FOUND:
            return "Timestamp sync file not found"
        elif code == ReturnCodes.DATASTREAM_FOLDER_NOT_FOUND:
            return "Datastream folder not found"
        elif code == ReturnCodes.BLOCKCHAIN_FILE_HEADER_WRONG:
            return "Blockchain file header wrong"
        elif code == ReturnCodes.FILTER_TYPE_INVALID:
            return "Invalid filter type"
        elif code == ReturnCodes.DATA_TYPE_INVALID:
            return "Invalid data type"
        elif code == ReturnCodes.CORE_ID_INVALID:
            return "Invalid core id"
        elif code == ReturnCodes.FILTER_ID_INVALID:
            return "Invalid filter id"
        elif code == ReturnCodes.INVALID_BLOCKCHAIN_LINE:
            return "Invalid line in blockchain"
        elif code == ReturnCodes.OBJECT_CLASSES_NOT_FOUND:
            return "Object classes file not found"
        elif code == ReturnCodes.OBJECT_CLASSES_INVALID:
            return "Invalid object classes file"

        elif code == ReturnCodes.INVALID_CALIBRATION_FILE_EXTENSION:
            return "Invalid calibration file extension"
        elif code == ReturnCodes.DB_PATH_NOT_DIR:
            return "DB path must be a directory not a file"
        elif code == ReturnCodes.CALIBRATION_FILE_NOT_FILE:
            return "Calibration path must be a file not a directory"
        elif code == ReturnCodes.CALIBRATION_FILE_NOT_FOUND:
            return "Calibration file not found"
        elif code == ReturnCodes.DB_PATH_NOT_FOUND:
            return "DB path not found"
        elif code == ReturnCodes.OBJECT_CLASSES_NOT_FILE:
            return "Object classes path must be a file not a directory"
        elif code == ReturnCodes.INVALID_OBJECT_CLASSES_FILE_EXTENSION:
            return "Invalid object classes file extension"

        return "Unknown error"


class CalibrationFileType(object):
    CAMERA = 1
    LIDAR = 2


def validate_datastream_folder(db_path, config_json):
    if not os.path.exists(db_path):
        return ReturnCodes.DB_PATH_NOT_FOUND

    if not os.path.isdir(db_path):
        return ReturnCodes.DB_PATH_NOT_DIR

    blockchain_descriptor_file = os.path.join(db_path, "datablock_descriptor.csv")
    if not os.path.exists(blockchain_descriptor_file):
        return ReturnCodes.BLOCKCHAIN_NOT_FOUND

    ts_sync_file = os.path.join(db_path, "sampling_timestamps_sync.csv")
    if not os.path.exists(ts_sync_file):
        return ReturnCodes.TIMESTAMP_SYNC_NOT_FOUND

    with open(blockchain_descriptor_file, "r") as blockchain_desc:
        lines_blockchain = blockchain_desc.readlines()

    header = lines_blockchain[0].strip().split(",")

    header_len = len(header)
    if "vision_core_id" not in header:
        return ReturnCodes.BLOCKCHAIN_FILE_HEADER_WRONG
    if "filter_id" not in header:
        return ReturnCodes.BLOCKCHAIN_FILE_HEADER_WRONG
    if "name" not in header:
        return ReturnCodes.BLOCKCHAIN_FILE_HEADER_WRONG
    if "type" not in header:
        return ReturnCodes.BLOCKCHAIN_FILE_HEADER_WRONG
    if "output_data_type" not in header:
        return ReturnCodes.BLOCKCHAIN_FILE_HEADER_WRONG
    if "input_sources" not in header:
        return ReturnCodes.BLOCKCHAIN_FILE_HEADER_WRONG
    for idx in range(1, len(lines_blockchain)):
        line_split = lines_blockchain[idx].strip().split(",")
        if len(line_split) != header_len:
            return ReturnCodes.INVALID_BLOCKCHAIN_LINE
        try:
            core_id = int(line_split[0])
        except ValueError:
            return ReturnCodes.CORE_ID_INVALID
        try:
            filter_id = int(line_split[1])
        except ValueError:
            return ReturnCodes.FILTER_TYPE_INVALID
        name = line_split[2]
        try:
            type = int(line_split[3])
        except ValueError:
            return ReturnCodes.INVALID_BLOCKCHAIN_LINE
        try:
            output_data_type = int(line_split[4])
        except ValueError:
            return ReturnCodes.INVALID_BLOCKCHAIN_LINE

        filter_type_valid = check_filter_type(type, config_json)
        if filter_type_valid != ReturnCodes.OK:
            return filter_type_valid

        output_data_type_valid = check_output_type(output_data_type, config_json)
        if output_data_type_valid != ReturnCodes.OK:
            return output_data_type_valid
        if not os.path.exists(os.path.join(db_path, "datastream_{}".format(filter_id))):
            return ReturnCodes.DATASTREAM_FOLDER_NOT_FOUND

    return ReturnCodes.OK


def validate_config(config_json):
    with open(config_json, "r") as jf:
        try:
            config_json = json.load(jf)
        except json.decoder.JSONDecodeError:
            return ReturnCodes.INVALID_CONFIG_JSON
    if "version" not in config_json:
        return ReturnCodes.INVALID_CONFIG_JSON
    if "date" not in config_json:
        return ReturnCodes.INVALID_CONFIG_JSON
    if "CyC_DataTypes" not in config_json:
        return ReturnCodes.INVALID_CONFIG_JSON
    for data_type in config_json["CyC_DataTypes"]:
        if "name" not in data_type:
            return ReturnCodes.INVALID_CONFIG_JSON
        if "value" not in data_type:
            return ReturnCodes.INVALID_CONFIG_JSON

    if "CyC_FilterTypes" not in config_json:
        return ReturnCodes.INVALID_CONFIG_JSON
    for filter_type in config_json["CyC_FilterTypes"]:
        if "name" not in filter_type:
            return ReturnCodes.INVALID_CONFIG_JSON
        if "value" not in filter_type:
            return ReturnCodes.INVALID_CONFIG_JSON
    return ReturnCodes.OK


def validate_calibration_file(calib_file_path, calibration_type):
    if not os.path.exists(calib_file_path):
        return ReturnCodes.CALIBRATION_FILE_NOT_FOUND

    if not os.path.isfile(calib_file_path):
        return ReturnCodes.CALIBRATION_FILE_NOT_FILE

    if not calib_file_path.split(".")[-1] == "cal":
        return ReturnCodes.INVALID_CALIBRATION_FILE_EXTENSION

    with io.open(calib_file_path) as f:
        try:
            sections = libconf.load(f)
        except libconf.ConfigParseError:
            return ReturnCodes.INVALID_CALIBRATION_FILE
    if "Pose" not in sections:
        return ReturnCodes.INVALID_CALIBRATION_FILE
    if "Rotation" not in sections["Pose"]:
        return ReturnCodes.INVALID_CALIBRATION_FILE
    if "x" not in sections["Pose"]["Rotation"]:
        return ReturnCodes.INVALID_CALIBRATION_FILE
    if "y" not in sections["Pose"]["Rotation"]:
        return ReturnCodes.INVALID_CALIBRATION_FILE
    if "z" not in sections["Pose"]["Rotation"]:
        return ReturnCodes.INVALID_CALIBRATION_FILE

    if "Translation" not in sections["Pose"]:
        return ReturnCodes.INVALID_CALIBRATION_FILE
    if "x" not in sections["Pose"]["Translation"]:
        return ReturnCodes.INVALID_CALIBRATION_FILE
    if "y" not in sections["Pose"]["Translation"]:
        return ReturnCodes.INVALID_CALIBRATION_FILE
    if "z" not in sections["Pose"]["Translation"]:
        return ReturnCodes.INVALID_CALIBRATION_FILE

    if calibration_type == CalibrationFileType.CAMERA:
        if "image_width" not in sections:
            return ReturnCodes.INVALID_CALIBRATION_FILE
        if "image_height" not in sections:
            return ReturnCodes.INVALID_CALIBRATION_FILE
        if "LeftSensor" not in sections:
            return ReturnCodes.INVALID_CALIBRATION_FILE
        if "channels" not in sections["LeftSensor"]:
            return ReturnCodes.INVALID_CALIBRATION_FILE
        if "focal_length_x" not in sections["LeftSensor"]:
            return ReturnCodes.INVALID_CALIBRATION_FILE
        if "focal_length_y" not in sections["LeftSensor"]:
            return ReturnCodes.INVALID_CALIBRATION_FILE
        if "optical_center_x" not in sections["LeftSensor"]:
            return ReturnCodes.INVALID_CALIBRATION_FILE
        if "optical_center_y" not in sections["LeftSensor"]:
            return ReturnCodes.INVALID_CALIBRATION_FILE
        if "pixel_size_x" not in sections["LeftSensor"]:
            return ReturnCodes.INVALID_CALIBRATION_FILE
        if "pixel_size_y" not in sections["LeftSensor"]:
            return ReturnCodes.INVALID_CALIBRATION_FILE
    return ReturnCodes.OK


def validate_object_classes_file(obj_classes_path):
    if not os.path.exists(obj_classes_path):
        return ReturnCodes.OBJECT_CLASSES_NOT_FOUND

    if not os.path.isfile(obj_classes_path):
        return ReturnCodes.OBJECT_CLASSES_NOT_FILE

    if not obj_classes_path.split(".")[-1] == "conf":
        return ReturnCodes.INVALID_OBJECT_CLASSES_FILE_EXTENSION

    with io.open(obj_classes_path) as f:
        try:
            classes = libconf.load(f)
        except libconf.ConfigParseError:
            return ReturnCodes.OBJECT_CLASSES_INVALID
        if "ObjectClasses" not in classes:
            return ReturnCodes.OBJECT_CLASSES_INVALID
        for obj_class in classes['ObjectClasses']:
            if "ID" not in classes["ObjectClasses"][obj_class]:
                return ReturnCodes.OBJECT_CLASSES_INVALID
            if "Countable" not in classes["ObjectClasses"][obj_class]:
                return ReturnCodes.OBJECT_CLASSES_INVALID
    return ReturnCodes.OK


def find_calibration_dependent_filters(db_path, config_json):
    log_signals = LogSignals()
    db_ok = validate_datastream_folder(db_path, config_json)
    lidars = list()
    cameras = list()
    if db_ok != ReturnCodes.OK:
        log_signals.error_signal.emit(db_ok, ReturnCodes.code2string(db_ok))
    else:
        blockchain_descriptor_file = os.path.join(db_path, "datablock_descriptor.csv")
        with open(blockchain_descriptor_file, "r") as blockchain_desc:
            lines = blockchain_desc.readlines()
        for idx in range(1, len(lines)):
            line_split = lines[idx].strip().split(",")
            core_id = int(line_split[0])
            filter_id = int(line_split[1])
            name = line_split[2]
            type = int(line_split[3])
            output_data_type = int(line_split[4])
            for filter_type in config_json["CyC_FilterTypes"]:
                if filter_type["value"] == type:
                    if filter_type["name"] == "CyC_MONO_CAMERA_FILTER_TYPE":
                        cameras.append({"name": name, "filter_id": filter_id})
                    elif filter_type["name"] == "CyC_LIDAR_FILTER_TYPE":
                        lidars.append({"name": name, "filter_id": filter_id})
    return cameras, lidars


def check_filter_type(type, config_json):
    for filter_type in config_json["CyC_FilterTypes"]:
        if filter_type["value"] == type:
            return ReturnCodes.OK
    return ReturnCodes.FILTER_TYPE_INVALID


def check_output_type(data_type, config_json):
    for d_type in config_json["CyC_DataTypes"]:
        if d_type["value"] == data_type:
            return ReturnCodes.OK
    return ReturnCodes.DATA_TYPE_INVALID


def test_config_json(config_json):
    if not os.path.exists(config_json):
        return ReturnCodes.CONFIG_NOT_FOUND
    config_valid = validate_config(config_json)
    return config_valid


def exit_on_fail(ret_code):
    if ret_code == ReturnCodes.OK:
        pass
    else:
        print("Error occured: {}".format(ReturnCodes.code2string(ret_code)))
        exit(ret_code)


def validate_paths(args):
    exit_on_fail(test_config_json(args.config_file))
    exit_on_fail(validate_calibration_file(args.camera_calibration_file, CalibrationFileType.CAMERA))
    exit_on_fail(validate_calibration_file(args.lidar_calibration_file, CalibrationFileType.LIDAR))
    exit_on_fail(validate_datastream_folder(args.datastream_path, json.load(open(args.config_file, "r"))))
    exit_on_fail(validate_object_classes_file(args.object_classes_file))


if __name__ == "__main__":
    args = Namespace(
        config_file="config.json",
        camera_calibration_file=r"C:\dev\CyberCortex.AI\core\etc\calibration\scout\quadcam_front_3.cal",
        lidar_calibration_file=r"C:\dev\CyberCortex.AI\core\etc\calibration\scout\hessai_lidar.cal",
        datastream_path=r'C:\data\scout_data\test_3',
        object_classes_file=r'C:\dev\CyberCortex.AI\core\etc\env\object_classes_mapillary.conf'
    )
    validate_paths(args)

    cameras, lidars = find_calibration_dependent_filters(db_path=args.datastream_path,
                                                         config_json=json.load(open(args.config_file, "r")))
    for camera in cameras:
        print("Camera {}, filter_id {}".format(
            camera["name"],
            camera["filter_id"]
        ))
    for lidar in lidars:
        print("Lidar {}, filter_id {}".format(
            lidar["name"],
            lidar["filter_id"]
        ))
