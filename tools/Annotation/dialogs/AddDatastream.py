"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing,
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from PyQt5.QtWidgets import QDialog, QSpinBox, QPushButton, QLabel, QLineEdit, QComboBox, QApplication
from PyQt5.QtCore import pyqtSignal
from PyQt5 import uic

from tools.Annotation.dialogs.FilterDatastreamErrorCodes import DatastreamErrorCodes
from data.types_CyC_TYPES import CyC_FilterType, CyC_DataType

from global_config import *
import sys
import csv


sys.path.insert(1, '../')
current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")
CFG = cfg


########################################################################################################################
# IN PROGRESS ##########################################################################################################
########################################################################################################################
class AddDatastreamWindow(QDialog):
    signal_process_finished = pyqtSignal(int)

    def __init__(self,):
        super().__init__()
        uic.loadUi(os.path.join(ui_directory, "AddDatastreamWindow.ui"), self)
        self.setWindowTitle("Add new datastream")

        # Widgets
        self.qtTitleLabel = self.findChild(QLabel, "qtTitleLabel")
        self.qtFilterNameLineEdit = self.findChild(QLineEdit, "qtFilterNameLineEdit")

        self.qtFilterIDSpinBox = self.findChild(QSpinBox, "qtFilterIDSpinBox")
        self.qtOverlayFilterIDSpinBox = self.findChild(QSpinBox, "qtOverlayFilterIDSpinBox")
        self.qtStartPushButton = self.findChild(QPushButton, "qtStartPushButton")
        self.qtFilterTypeComboBox = self.findChild(QComboBox, "qtFilterTypeComboBox")

        self.filter_types = {"Semantic_Segmentation":
                                 ["timestamp_stop,sampling_time,semantic,instances\n",
                                  "timestamp_stop,shape_id,cls,instance,points\n"],
                             "2D":
                                 ["timestamp_stop,sampling_time,frame_id\n",
                                  "frame_id,roi_id,cls,x,y,width,height\n"],
                             "3D":
                                 ["timestamp_stop,sampling_time,frame_id\n",
                                  "frame_id,roi_id,cls,x,y,z,w,h,l,roll,pitch,yaw\n"],
                             "Points_Tracker":
                                 ["timestamp_stop,sampling_time,frame_id\n",
                                  "frame_id,x,y,id,score\n"],
                             "Lane_Detection":
                                 ["timestamp_stop,sampling_time,semantic,instances\n",
                                  "timestamp_stop,lane_id,points,theta_0,theta_1,theta_2,theta_3\n"]
        }
        self.qtFilterTypeComboBox.addItems(self.filter_types.keys())

        # Signals
        self.qtStartPushButton.clicked.connect(self.on_start_button)
        self.qtFilterTypeComboBox.currentIndexChanged.connect(self.on_filter_type_changed)

        self.qtFilterTypeComboBox.setCurrentIndex(1)

    # Slots ------------------------------------------------------------------------------------------------------------
    def on_filter_type_changed(self):
        self.qtTitleLabel.setText("Add {} datastream".format(self.qtFilterTypeComboBox.currentText()))
        self.qtFilterNameLineEdit.setText("{}_Datastream".format(self.qtFilterTypeComboBox.currentText()))

    def on_start_button(self):
        res = AddDatastreamWindow.init_datastream(
            stream_name=self.findChild(QLineEdit, "qtFilterNameLineEdit").text(),
            base_path=CFG.DB.BASE_PATH,
            filter_id=self.qtFilterIDSpinBox.value(),
            overlay_filter_id=self.qtOverlayFilterIDSpinBox.value(),
            filter_headers=self.filter_types[self.qtFilterTypeComboBox.currentText()]
        )

        self.signal_process_finished.emit(res)
        self.close()

    # Static Methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def init_datastream(stream_name, base_path, filter_id, overlay_filter_id, filter_headers):
        if not os.path.exists(base_path):
            print("Base path {} does not exist. Could not create new datastream.".format(base_path))
            return

        ds_path = os.path.join(base_path, "datastream_{}".format(filter_id))

        #  Create base folder
        if not os.path.exists(ds_path):
            os.mkdir(ds_path)

        res = AddDatastreamWindow._create_data_descriptor(ds_path, base_path, overlay_filter_id, filter_headers)
        if res != DatastreamErrorCodes.OK:
            return res

        AddDatastreamWindow._create_framebased_data_descriptor(ds_path, filter_headers)

        res = AddDatastreamWindow._update_data_block_descriptor(stream_name, base_path, filter_id, overlay_filter_id)
        if res != DatastreamErrorCodes.OK:
            return res

        res = AddDatastreamWindow._update_sampling_timestamps_sync(base_path, filter_id, overlay_filter_id)
        if res != DatastreamErrorCodes.OK:
            return res

        if len(filter_headers[0].split(',')) != 4:
            # Create samples folder
            if not os.path.exists(os.path.join(ds_path, "samples")):
                os.mkdir(os.path.join(ds_path, "samples"))
            if not os.path.exists(os.path.join(ds_path, "samples", "0")):
                os.mkdir(os.path.join(ds_path, "samples", "0"))
            if not os.path.exists(os.path.join(ds_path, "samples", "0", "left")):
                os.mkdir(os.path.join(ds_path, "samples", "0", "left"))

            if not os.path.exists(os.path.join(ds_path, "samples", "1")):
                os.mkdir(os.path.join(ds_path, "samples", "1"))
            if not os.path.exists(os.path.join(ds_path, "samples", "1", "left")):
                os.mkdir(os.path.join(ds_path, "samples", "1", "left"))

        return DatastreamErrorCodes.OK

    @staticmethod
    def _create_data_descriptor(ds_path, base_path, overlay_filter_id, filter_headers):
        # Create data_descriptor file
        descriptor_file = os.path.join(ds_path, "data_descriptor.csv")

        if not os.path.exists(descriptor_file):
            overlay_descriptor_path = os.path.join(base_path, "datastream_{}".format(overlay_filter_id),
                                                   "data_descriptor.csv")
            if not os.path.exists(overlay_descriptor_path):
                return DatastreamErrorCodes.OVERLAY_DATASTREAM_NOT_FOUND

            #  Write ts_start, ts_stop, sampling time from image overlay stream
            #  and write frame_id as increasing counter
            with open(overlay_descriptor_path, "r") as overlay:
                with open(descriptor_file, "w") as desc_f:
                    desc_f.write(filter_headers[0])
                    overlay_reader = csv.DictReader(overlay)
                    idx = 0
                    for row in overlay_reader:
                        aux = str(idx)

                        if len(filter_headers[0].split(',')) != 4:
                            aux = ","

                        desc_f.write("{},{},{}\n".format(
                            row["timestamp_stop"],
                            row["sampling_time"],
                            aux
                        ))
                        idx += 1

        return DatastreamErrorCodes.OK

    @staticmethod
    def _create_framebased_data_descriptor(ds_path, filter_headers):
        # Create framebased_data_descriptor file
        framebased_descriptor_file = os.path.join(ds_path, "framebased_data_descriptor.csv")
        if not os.path.exists(framebased_descriptor_file):
            with open(framebased_descriptor_file, "w") as f_desc_f:
                f_desc_f.write(filter_headers[1])

    @staticmethod
    def _update_data_block_descriptor(stream_name, base_path, filter_id, overlay_filter_id):
        # Check if datastream is in datablock_descriptor.csv
        blockchain_file = os.path.join(base_path, "datablock_descriptor.csv")
        if not os.path.exists(blockchain_file):
            print("ERROR: Blockchain file {} not found. Could not create semantic segmentation datastream".format(
                blockchain_file
            ))
            return DatastreamErrorCodes.BLOCKCHAIN_FILE_NOT_FOUND

        found_datastream = False
        with open(blockchain_file, "r") as blockchain_desc:
            reader = csv.DictReader(blockchain_desc)
            for row in reader:
                if int(row["filter_id"]) == filter_id:
                    found_datastream = True
                    break

        # Add datastream to datablock_descriptor
        if found_datastream is False:
            with open(blockchain_file, "r") as blockchain_desc:
                lines = blockchain_desc.readlines()

            DEFAULT_CORE_ID = 1
            lines.append("{0},{1},{2},{3},{4},{{{5}-{6}}}".format(
                DEFAULT_CORE_ID,
                filter_id,
                stream_name,
                CyC_FilterType.CyC_OBJECT_DETECTOR_2D_FILTER_TYPE,
                CyC_DataType.CyC_2D_ROIS,
                DEFAULT_CORE_ID,
                overlay_filter_id
            ))

            with open(blockchain_file, "w") as blockchain_desc:
                for line in lines:
                    blockchain_desc.write("{}\n".format(line.strip()))
        else:
            print("ERROR: Datastream with filter_id={} already present in database".format(filter_id))
            return DatastreamErrorCodes.DATASTREAM_ALREADY_PRESENT

        return DatastreamErrorCodes.OK

    @staticmethod
    def _update_sampling_timestamps_sync(base_path, filter_id, overlay_filter_id):
        # Check if datastream is in sampling_timestamps_sync.csv file
        ts_sync_file = os.path.join(base_path, "sampling_timestamps_sync.csv")
        if not os.path.exists(ts_sync_file):
            print("ERROR: Timestamp sync file {} not found. Could not create semantic segmentation datastream".format(
                ts_sync_file
            ))

        found_datastream = False
        with open(ts_sync_file, "r") as ts_sync:
            reader = csv.DictReader(ts_sync)
            for row in reader:
                if "datastream_{}".format(filter_id) in row:
                    found_datastream = True
                    break

        if not found_datastream:
            with open(ts_sync_file, "r") as ts_sync:
                lines = ts_sync.readlines()
            # Add datastream to header
            lines[0] = lines[0].strip(",").strip("\n") + ",datastream_{}".format(filter_id)
            try:
                header_values = lines[0].split(",")
                header_values = [h.strip() for h in header_values]
                overlay_index = header_values.index("datastream_{}".format(overlay_filter_id))
            except ValueError:
                print("ERROR: Overlay datastream not found in timesync file")
                return DatastreamErrorCodes.OVERLAY_DATASTREAM_NOT_FOUND

            # Add datastream to each line in timesync file
            for idx in range(1, len(lines)):
                overlay_ts = lines[idx].split(",")[overlay_index]
                lines[idx] = lines[idx].strip(",").strip("\n") + ",{}".format(overlay_ts)
            with open(ts_sync_file, "w") as ts_sync:
                for line in lines:
                    ts_sync.write("{}\n".format(line.strip()))

        return DatastreamErrorCodes.OK


def main():
    try:
        app = QApplication([])
        window = AddDatastreamWindow()
        window.show()
        app.exec_()
    except RuntimeError:
        exit(0)


if __name__ == "__main__":
    main()
