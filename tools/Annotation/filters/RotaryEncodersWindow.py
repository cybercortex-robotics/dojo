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


class RotaryEncodersWindow(FilterWindowInterface):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id):
        super().__init__(filter_id, is_static=True)
        uic.loadUi(os.path.join(ui_directory, 'PlotWindow.ui'), self)

        # Widget
        self.find_children()

        # Members
        self.states = list()
        self.graphics = None

        # Initialization
        self.init_window()
        self.init_splitter()

    # Inherited Initialization Methods ---------------------------------------------------------------------------------
    def init_window(self):
        layout = QGridLayout()
        self.graphics = MplCanvas(self)
        self.qtPlotWidget.setLayout(layout)
        self.qtPlotWidget.layout().addWidget(self.graphics)

    # Private Methods --------------------------------------------------------------------------------------------------
    def _get_data(self, idx):
        data = list()
        for i in range(len(self.states)):
            data.append(self.states[i][idx])

        return data

    def _draw_graphic(self, idx):
        data = self._get_data(idx)
        self.graphics.axes.grid(True, linestyle='--')
        self.graphics.axes.plot(data, linewidth=2, label="data_{}".format(idx))
        self.graphics.axes.legend()
        self.graphics.axes.set_xlabel('Time [ms]')
        self.graphics.axes.set_ylabel('Increment [ticks]')

    # Inherited Slots Load Data ----------------------------------------------------------------------------------------
    def on_load_all_data(self):
        data_stream = self.parser.get_data_by_id(self.filter_id)

        if data_stream is None:
            self.signal_error_signal.emit("Datastream NONE!", self.filter_id)
            return

        for ts_stop, sampling_time, data in data_stream:
            self.states.append(data)
            self.show_runtime_values(ts_stop=ts_stop, ts_sampling_time=sampling_time)

        if len(self.states) == 0:
            self.signal_warning_signal.emit("Datastream is empty!", self.filter_id)
            return

        self.graphics.axes.cla()
        for idx in range(len(self.states[0])):
            self._draw_graphic(idx)
