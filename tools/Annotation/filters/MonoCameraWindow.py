"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from tools.Annotation.filters.FilterWindowInterface import FilterWindowInterface

from PyQt5.QtWidgets import *
from PyQt5 import uic

import os

current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")


class MonoCameraWindow(FilterWindowInterface):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id):
        super().__init__(filter_id)
        uic.loadUi(os.path.join(ui_directory, 'MonoCameraWindow.ui'), self)

        # Widget
        self.find_children()

        # Initialization
        self.scene = QGraphicsScene()
        self.qtGraphicsView.setScene(self.scene)

        self.init_splitter()
        self.show_calibration_file()

    # Inherited Slots Draw/Delete/Save ---------------------------------------------------------------------------------
    def on_click_button_save(self):
        self.save_calibration_file()

    # Inherited Slots Load Data ----------------------------------------------------------------------------------------
    def on_load_data(self, line_index, ts_sync_timestamp):
        data = self.parser.parse_line(filter_id=self.filter_id, line_index=line_index)
        self.frame_id = line_index - 1

        if not self.data_is_valid(data):
            self.signal_error_signal.emit("Invalid data frame id {}, line index {}, timestamp sync {}!"
                                          .format(self.frame_id, line_index, ts_sync_timestamp), self.filter_id)
            return

        ts_stop, sampling_time, ts_image, cv_image, _ = data
        self.show_runtime_values(frame_id=self.frame_id, ts_stop=ts_stop,
                                 ts_sampling_time=sampling_time, ts_image=ts_image)
        self.load_image(cv_image=cv_image)
        self.on_main_splitter_moved()
