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

from toolkit.vision.plot_utils import MplCanvas

from PyQt5.QtWidgets import QGridLayout
from PyQt5 import uic

import os

current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")


class ImuWindow(FilterWindowInterface):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id):
        super().__init__(filter_id, is_static=True)
        uic.loadUi(os.path.join(ui_directory, 'PlotWindow.ui'), self)

        # Widget
        self.find_children()

        # Members
        self.graphics = list()
        self.labels = dict()
        self.all_data = dict()

        # Initialization
        self.init_window()
        self.init_splitter()

    # Inherited Initialization Methods ---------------------------------------------------------------------------------
    def init_window(self):
        # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        self.labels[0] = ['acc_x', 'acc_y', 'acc_z']
        self.labels[1] = ['gyro_x', 'gyro_y', 'gyro_z']
        self.labels[2] = ['ts_stop']

        for key, value in self.labels.items():
            for label in value:
                self.all_data[label] = list()

        self.graphics.append(MplCanvas(self))
        self.graphics.append(MplCanvas(self))

        layout = QGridLayout()
        self.qtPlotWidget.setLayout(layout)
        self.qtPlotWidget.layout().addWidget(self.graphics[0])
        self.qtPlotWidget.layout().addWidget(self.graphics[1])

    # Inherited Slots Load Data ----------------------------------------------------------------------------------------
    def on_load_all_data(self):
        data_stream = self.parser.get_data_by_id(self.filter_id)
        data_stream = list(data_stream)

        if data_stream is None:
            self.signal_error_signal.emit("Datastream NONE!", self.filter_id)
            return

        if len(data_stream) == 0:
            self.signal_error_signal.emit("Datastream is empty!", self.filter_id)
            return

        for ts_stop, sampling_time, data in data_stream:
            for d in data:
                state, time = d
                self.all_data['acc_x'].append(state[0])
                self.all_data['acc_y'].append(state[1])
                self.all_data['acc_z'].append(state[2])
                self.all_data['gyro_x'].append(state[3])
                self.all_data['gyro_y'].append(state[4])
                self.all_data['gyro_z'].append(state[5])
                self.all_data['ts_stop'].append(ts_stop)
                self.show_runtime_values(ts_stop=ts_stop, ts_sampling_time=sampling_time)

        if len(self.all_data['ts_stop']) == 0:
            self.signal_warning_signal.emit("Datastream is empty!", self.filter_id)

        self._draw_graphic(0)  # acc
        self._draw_graphic(1)  # gyro

    # Private Methods --------------------------------------------------------------------------------------------------
    def _draw_graphic(self, idx):
        labels = self.labels[idx]
        self.graphics[idx].axes.grid(True, linestyle='--')
        self.graphics[idx].axes.plot(self.all_data['ts_stop'], self.all_data[labels[0]], color='red',
                                     linewidth=2, label=labels[0])
        self.graphics[idx].axes.plot(self.all_data['ts_stop'], self.all_data[labels[1]], color='green',
                                     linewidth=2, label=labels[1])
        self.graphics[idx].axes.plot(self.all_data['ts_stop'], self.all_data[labels[2]], color='blue',
                                     linewidth=2, label=labels[2])
        self.graphics[idx].axes.legend()
        self.graphics[0].axes.set_ylabel('Acceleration [m/s^2]')
        self.graphics[1].axes.set_ylabel('Gyroscope [rad/s]')
        self.graphics[1].axes.set_xlabel('Time [ms]')
