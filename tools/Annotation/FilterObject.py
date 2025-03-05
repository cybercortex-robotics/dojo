"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing,
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from PyQt5.QtWidgets import QTreeWidgetItem, QMessageBox

from tools.Annotation.filters.ImuWindow import ImuWindow
from tools.Annotation.filters.LanesWindow import LanesWindow
from tools.Annotation.filters.MonoCameraWindow import MonoCameraWindow
from tools.Annotation.filters.LidarWindow import LidarWindow
from tools.Annotation.filters.PointsTrackingWindow import PointsTrackingWindow
from tools.Annotation.filters.Roi2DWindow import Roi2DWindow
from tools.Annotation.filters.Roi3DWindow import Roi3DWindow
from tools.Annotation.filters.RotaryEncodersWindow import RotaryEncodersWindow
from tools.Annotation.filters.SemSegWindow import SemSegWindow
from tools.Annotation.filters.StateEstimationWindow import StateEstimationWindow
from tools.Annotation.filters.StereoCameraWindow import StereoCameraWindow

from data.types_CyC_TYPES import CyC_FilterType
from data.CyC_db_interface import filtertype2string, filteroutput2string
from global_config import *

import pandas as pd
import csv
import shutil

CFG = cfg
current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "ui")


class TreeWidgetItemColumns(object):
    DELETE_ICON = 0
    FILTER_CORE = 1
    FILTER_ID = 2
    FILTER_NAME = 3


class FilterWidgetItem(QTreeWidgetItem):
    def __init__(self, _filter, parent, **args):
        super(FilterWidgetItem, self).__init__(parent, **args)
        self.filter = _filter

    def get_filter(self):
        return self.filter

    def get_window(self):
        return self.filter.get_window()

    def change_filter_id(self):
        self.filter.set_filter_id(self.text(TreeWidgetItemColumns.FILTER_ID))

    def change_filter_name(self):
        self.filter.set_filter_name(self.text(TreeWidgetItemColumns.FILTER_NAME))

    def delete_filter(self):
        self.filter.delete()


########################################################################################################################
# IN PROGRESS ##########################################################################################################
########################################################################################################################
class FilterObject:
    def __init__(self, row, control_window, parser):
        self.filter_core_id = int(row["vision_core_id"])
        self.filter_id = int(row["filter_id"])
        self.filter_name = row["name"]
        self.filter_type = int(row["type"])
        self.filter_output_data_type = int(row["output_data_type"])

        self.input_sources = "NO"
        if row["input_sources"] is not None:
            self.input_sources = row["input_sources"]

        self.parent = control_window
        self.parser = parser
        self.window = self._create_window()

    # METHODS ----------------------------------------------------------------------------------------------------------
    @staticmethod
    def _change_data_descriptor_file(old_line, new_line=""):
        datablock_descriptor_file = os.path.join(CFG.DB.BASE_PATH, 'datablock_descriptor.csv')

        with open(datablock_descriptor_file, 'r', encoding='utf-8') as file:
            content = file.readlines()

            if new_line == "":
                content.remove(old_line)
            else:
                content[content.index(old_line)] = new_line

        with open(datablock_descriptor_file, 'w', encoding='utf-8') as file:
            file.writelines(content)

    @staticmethod
    def _is_new_filter_id_valid(_id):
        datablock_descriptor_file = os.path.join(CFG.DB.BASE_PATH, 'datablock_descriptor.csv')
        with open(datablock_descriptor_file, 'r', newline='') as csv_file:
            for row in csv.reader(csv_file):
                try:
                    filter_id = int(row[1])
                    if filter_id == int(_id):
                        return False
                except Exception:
                    pass
        return True

    def _remove_filter_directory(self):
        try:
            self.parser.close_filter_parser_by_id(self.filter_id)
            shutil.rmtree(os.path.join(CFG.DB.BASE_PATH, 'datastream_{}'.format(self.get_filter_id())),
                          ignore_errors=False)
            self.parent.parent.on_display_info("Datastream {} was deleted!".format(self.filter_id))
        except PermissionError:
            self.parent.parent.on_display_error("Failed to delete all files of datastream {}".format(self.filter_id))
            raise

    def _change_directory_name(self, old_filter_id):
        source = os.path.join(CFG.DB.BASE_PATH, 'datastream_{}'.format(old_filter_id))
        destination = os.path.join(CFG.DB.BASE_PATH, 'datastream_{}'.format(self.get_filter_id()))

        try:
            self.parser.close_filter_parser_by_id(int(old_filter_id))
            shutil.move(source, destination)
            self.parent.parent.on_display_info(
                    "Datastream {} directory was renamed to {}!".format(old_filter_id, self.get_filter_id()))
        except PermissionError:
            self.parent.parent.on_display_error("Failed to rename datastream {} directory!".format(old_filter_id))
            raise

    def _change_sampling_timestamps_sync(self, old_filter_id, remove=False):
        sampling_timestamps_sync_file = os.path.join(CFG.DB.BASE_PATH, 'sampling_timestamps_sync.csv')
        data = pd.read_csv(sampling_timestamps_sync_file)

        if remove:
            data.drop(['datastream_{}'.format(old_filter_id)], axis=1, inplace=True)
        else:
            data.rename(columns={'datastream_{}'.format(old_filter_id): 'datastream_{}'.format(self.get_filter_id())},
                        inplace=True)

        data.to_csv(sampling_timestamps_sync_file, index=False)

    def _is_filter_dependency(self):
        datablock_descriptor_file = os.path.join(CFG.DB.BASE_PATH, 'datablock_descriptor.csv')

        with open(datablock_descriptor_file, 'r', newline='') as csv_file:
            for row in csv.reader(csv_file):
                input_sources = row[5]

                if input_sources != "" and input_sources != "input_sources":
                    sources = input_sources[1:-1].split(';')

                    for source in sources:
                        dependent_filter_id = source.split("-")[1]

                        if int(dependent_filter_id) == int(self.filter_id):
                            return True
        return False

    def delete(self):
        if self._is_filter_dependency():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("Cannot delete filter {}! It is an input source for other filters!".format(self.filter_id))
            msg.exec()
            return

        result = QMessageBox.question(self.parent, "Confirm Exit...",
                                      "Are you sure you want to delete {} ?".format(self.filter_name),
                                      QMessageBox.Yes | QMessageBox.No)
        if result == QMessageBox.No:
            return

        self._change_data_descriptor_file(old_line=self.get_datablock_descriptor_line())
        self._change_sampling_timestamps_sync(old_filter_id=self.filter_id, remove=True)
        self._remove_filter_directory()

        self.parent.reload_db()

    def set_filter_name(self, _name):
        old_line = self.get_datablock_descriptor_line()

        self.filter_name = _name
        new_line = self.get_datablock_descriptor_line()

        self._change_data_descriptor_file(old_line, new_line)
        self.parent.reload_db()

        self.parent.parent.on_display_info(
            "Filter name was changed: {} -> {}".format(old_line.split(',')[2], _name), self.get_filter_id())

    def set_filter_id(self, _id):
        if self._is_new_filter_id_valid(_id):
            old_filter_id = self.get_filter_id()
            old_line = self.get_datablock_descriptor_line()

            self.filter_id = int(_id)
            new_line = self.get_datablock_descriptor_line()

            self._change_data_descriptor_file(old_line=old_line, new_line=new_line)
            self._change_directory_name(old_filter_id=old_filter_id)
            self._change_sampling_timestamps_sync(old_filter_id=old_filter_id)

            self.parent.parent.on_display_info(
                "Filter ID was changed: {} -> {}".format(old_filter_id, _id), self.filter_id)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("Cannot change filter id {}! Already exists filter ID {}!".format(self.filter_id, _id))
            msg.exec()

        self.parent.reload_db()

    # GETTERS ----------------------------------------------------------------------------------------------------------
    def get_detail(self):
        input_sources_string = str(self.input_sources).replace("\'", "").replace("1-", "")

        return "* Core ID: {}\n" \
               "* Filter ID: {}\n" \
               "* Filter name: {}\n" \
               "* Filter type: {}\n" \
               "* Filter output data type: {}\n" \
               "* Input sources: {}\n" \
            .format(self.filter_core_id, self.filter_id, self.filter_name, filtertype2string(self.filter_type),
                    filteroutput2string(self.filter_output_data_type), input_sources_string)

    def get_datablock_descriptor_line(self):
        return "{},{},{},{},{},{}\n".format(self.filter_core_id, self.filter_id, self.filter_name,
                                            self.filter_type, self.filter_output_data_type, self.input_sources)

    def get_filter_name(self):
        return self.filter_name

    def get_filter_id(self):
        return self.filter_id

    def get_filter_core_id(self):
        return self.filter_core_id

    def get_window(self):
        return self.window

    # METHODS ----------------------------------------------------------------------------------------------------------
    def _create_window(self):
        input_sources = self.input_sources[1:-1].split(";")
        window = None

        if self.filter_type == CyC_FilterType.CyC_MONO_CAMERA_FILTER_TYPE:  # 1-6
            window = MonoCameraWindow(self.filter_id)
        elif self.filter_type == CyC_FilterType.CyC_RGBDCAMERA_FILTER_TYPE:
            window = StereoCameraWindow(self.filter_id)
        elif self.filter_type == CyC_FilterType.CyC_VEHICLE_STATE_ESTIMATION_FILTER_TYPE:  # 13
            window = StateEstimationWindow(self.filter_id)
        elif self.filter_type == CyC_FilterType.CyC_LIDAR_FILTER_TYPE:  # 7
            window = LidarWindow(self.filter_id)
        elif self.filter_type == CyC_FilterType.CyC_IMU_FILTER_TYPE:  # 27
            window = ImuWindow(self.filter_id)
        elif self.filter_type == CyC_FilterType.CyC_ROTARY_ENCODER_FILTER_TYPE:  # 36
            window = RotaryEncodersWindow(self.filter_id)

        # TODO: IN PROGRESS
        elif self.filter_type == CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE:  # 34, 37
            window = SemSegWindow(self.filter_id, input_sources)
        elif self.filter_type == CyC_FilterType.CyC_OBJECT_DETECTOR_2D_FILTER_TYPE:  # 14-19
            window = Roi2DWindow(self.filter_id, input_sources)
        elif self.filter_type == CyC_FilterType.CyC_OBJECT_DETECTOR_ROAD_FILTER_TYPE:
            window = Roi2DWindow(self.filter_id, input_sources, road=True)
        elif self.filter_type == CyC_FilterType.CyC_OBJECT_DETECTOR_3D_FILTER_TYPE:  # 20-25
            window = Roi3DWindow(self.filter_id, input_sources)
        elif self.filter_type == CyC_FilterType.CyC_POINTS_TRACKER_FILTER_TYPE:  # 38
            window = PointsTrackingWindow(self.filter_id, input_sources)
        elif self.filter_type == CyC_FilterType.CyC_LANE_DETECTION_FILTER_TYPE:  # 32, 33
            window = LanesWindow(self.filter_id, input_sources)
        # elif self.filter_type == CyC_FilterType.CyC_ULTRASONICS_FILTER_TYPE:
        #     window = USonicsWindow(self.filter_id)
        # elif self.filter_type == CyC_FilterType.CyC_ULTRASONIC_PREDICTION_FILTER_TYPE:
        #     window = USonicPredWindow(self.filter_id)
        # elif self.filter_type == CyC_FilterType.CyC_GPS_FILTER_TYPE:
        #     window = GpsWindow(self.filter_id)

        if window is not None:
            window.signal_info_signal.connect(self.parent.parent.on_display_info)
            window.signal_warning_signal.connect(self.parent.parent.on_display_warning)
            window.signal_error_signal.connect(self.parent.parent.on_display_error)
            window.on_load_all_data()
        else:
            self.parent.parent.on_display_warning("Filter type: {} with filter id: {} not supported!"
                                                  .format(filtertype2string(self.filter_type), self.filter_id))

        return window
