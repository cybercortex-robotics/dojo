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

from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt
import numpy as np

import math
import os

current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")


class StateEstimationWindow(FilterWindowInterface):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id):
        super().__init__(filter_id, is_static=True)
        uic.loadUi(os.path.join(ui_directory, 'PlotWindow.ui'), self)

        # Widget
        self.find_children()

        # Members
        self.labels = ['x', 'y', 'velocity', 'yaw']
        self.states = dict()
        self.graphics = None

        # Initialization
        self.init_window()
        self.init_splitter()

    # Inherited Initialization Methods ---------------------------------------------------------------------------------
    def init_window(self):
        self.graphics = MplCanvas(self)
        for label in self.labels:
            self.states[label] = list()

        layout = QGridLayout()
        self.qtPlotWidget.setLayout(layout)
        self.qtPlotWidget.layout().addWidget(self.graphics)

    # Private Methods --------------------------------------------------------------------------------------------------
    def _color_line(self, x, y, z=None, c_map='copper', line_width=3, alpha=1.0):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(min(z), max(z))

        # Default colors equally spaced on [0,0.01]:
        if z is None:
            z = np.linspace(0.0, 0.01, len(x))

        # Special case if a single number:
        # to check for numerical input -- this is a hack
        if not hasattr(z, "__iter__"):
            z = np.array([z])

        z = np.asarray(z)
        lc = LineCollection(segments, array=z, cmap=c_map, norm=norm, linewidth=line_width, alpha=alpha)
        self.graphics.axes.add_collection(lc)

        return lc

    def _draw_graphic(self):
        line = self._color_line(self.states['x'], self.states['y'], self.states['velocity'], c_map='jet')
        self.graphics.fig.colorbar(line)
        self.graphics.axes.set_xlim(min(self.states['x'])-0.5, max(self.states['x'])+0.5)
        self.graphics.axes.set_ylim(min(self.states['y'])-0.5, max(self.states['y'])+0.5)
        self.graphics.axes.grid(True, linestyle='--')
        self.graphics.axes.set_title('Vehicle traveled path and state [x, y, velocity, heading]')
        self.graphics.axes.set_xlabel('x [m]')
        self.graphics.axes.set_ylabel('y [m]')

        for i in range(len(self.states['x'])):
            self.graphics.axes.plot(self.states['x'][i], self.states['y'][i], marker=(3, 0, self.states['yaw'][i]),
                                    markerfacecolor='darkblue', color='white', markersize=10, linewidth=2)

        self.graphics.draw()

    def _data_is_valid(self):
        for label in self.labels:
            if len(self.states[label]) == 0:
                self.signal_warning_signal("Datastream is empty! Filter id {}!".format(self.filter_id), self.filter_id)
                return False
            for st in self.states[label]:
                if st is None:
                    return False
        return True

    # Inherited Slots Load Data ----------------------------------------------------------------------------------------
    def on_load_all_data(self):
        data_stream = self.parser.get_data_by_id(self.filter_id)
        if data_stream is None:
            self.signal_error_signal.emit("Datastream is none! Filter id {}!".format(self.filter_id), self.filter_id)
            return

        for ts_stop, sampling_time, state in data_stream:
            self.states['x'].append(state[0])
            self.states['y'].append(state[1])
            self.states['velocity'].append(state[2])
            self.states['yaw'].append(math.degrees(state[3]))
            self.show_runtime_values(ts_stop=ts_stop, ts_sampling_time=sampling_time)

        if not self._data_is_valid():
            self.signal_error_signal.emit("Datastream is not valid! Filter id {}!".format(self.filter_id),
                                          self.filter_id)
            return

        self.graphics.axes.cla()
        self._draw_graphic()
