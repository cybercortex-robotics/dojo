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

import pyqtgraph.opengl as gl

import os

current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")


class LidarWindow(FilterWindowInterface):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id):
        super().__init__(filter_id)
        uic.loadUi(os.path.join(ui_directory, 'LidarWindow.ui'), self)

        # Widget
        self.find_children()
        self.qtGraphicsView = self.findChild(QWidget, 'qtGraphicsView')
        self.qtPointsNum = self.findChild(QLabel, 'qtPointsNum')

        # Members
        self.gl_view = gl.GLViewWidget()
        self.pcl = None

        # Initialization
        self.init_window()
        self.init_splitter()
        self.show_calibration_file()

    # Inherited Initialization Methods ---------------------------------------------------------------------------------
    def init_window(self):
        z_grid = gl.GLGridItem()
        z_grid.scale(10, 10, 10)
        self.gl_view.addItem(z_grid)

        axis = gl.GLAxisItem()
        self.gl_view.addItem(axis)
        self.gl_view.setCameraPosition(distance=300)

        layout = QGridLayout()
        self.qtGraphicsView.setLayout(layout)
        self.qtGraphicsView.layout().addWidget(self.gl_view)

    # Inherited Method -------------------------------------------------------------------------------------------------
    def clear(self):
        if self.pcl:
            self.gl_view.removeItem(self.pcl)

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

        ts_stop, sampling_time, lidar = data
        self.show_runtime_values(ts_stop=ts_stop, ts_sampling_time=sampling_time, frame_id=self.frame_id)

        self.clear()
        if len(lidar) == 0:
            return

        self.pcl = gl.GLScatterPlotItem(pos=lidar, color=(0, 1, 0, .3), size=0.5, pxMode=False)
        self.qtPointsNum.setText(str(len(lidar)))
        self.gl_view.addItem(self.pcl)
        self.on_main_splitter_moved()
