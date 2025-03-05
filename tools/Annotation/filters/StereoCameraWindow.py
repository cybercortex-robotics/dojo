"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import numpy as np
import os
import cv2

from tools.Annotation.filters.FilterWindowInterface import FilterWindowInterface
from toolkit.vision.image_utils import decode_depth

from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")


class StereoCameraWindow(FilterWindowInterface):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id):
        super().__init__(filter_id)
        uic.loadUi(os.path.join(ui_directory, 'StereoCameraWindow.ui'), self)

        # Widget
        self.find_children()
        self.qCamRight = self.findChild(QGraphicsView, 'qCamRight')
        self.qCamLeft = self.findChild(QGraphicsView, 'qCamLeft')

        # Initialization
        self.init_window()
        self.init_splitter()

    # Inherited Initialization Methods ---------------------------------------------------------------------------------
    def init_window(self):
        self._init_graphics_view(self.qCamRight)
        self._init_graphics_view(self.qCamLeft)

    # Static Methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _init_graphics_view(view):
        scene = QGraphicsScene()
        view.setScene(scene)
        view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    # Inherited Slots Load Data ----------------------------------------------------------------------------------------
    def on_load_data(self, line_index, ts_sync_timestamp):
        data = self.parser.parse_line(filter_id=self.filter_id, line_index=line_index)
        self.frame_id = line_index - 1

        if not self.data_is_valid(data):
            self.signal_error_signal.emit("Invalid data frame id {}, line index {}, timestamp sync {}!"
                                          .format(self.frame_id, line_index, ts_sync_timestamp), self.filter_id)
            return

        ts_stop, sampling_time, img_l, img_r = data
        self.show_runtime_values(ts_stop=ts_stop,
                                 ts_sampling_time=sampling_time, frame_id=self.frame_id)

        img_r = img_r[..., ::-1].copy()
        img_depth = decode_depth(img_r)
        img_depth = img_depth.copy() / img_depth.max()
        img_depth = cv2.applyColorMap(np.uint8(img_depth * 255), cv2.COLORMAP_JET)

        self.load_image_in_view(img_l, self.qCamLeft)
        self.load_image_in_view(img_depth, self.qCamRight)
        self.on_main_splitter_moved()

    # Inherited Slots --------------------------------------------------------------------------------------------------
    def on_main_splitter_moved(self):
        self.qCamLeft.fitInView(self.qCamLeft.scene().sceneRect(), Qt.KeepAspectRatio)
        self.qCamRight.fitInView(self.qCamRight.scene().sceneRect(), Qt.KeepAspectRatio)
