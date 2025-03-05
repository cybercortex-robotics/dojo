"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class DojoGraphicScene(QGraphicsScene):
    # Signals ----------------------------------------------------------------------------------------------------------
    signal_draw_new_roi = pyqtSignal(int, int, int, int)
    signal_draw_temp_roi = pyqtSignal(int, int, int, int)
    signal_cursor_position = pyqtSignal(int, int)
    signal_draw_new_lane = pyqtSignal(int, int)
    signal_draw_new_point = pyqtSignal(int, int)
    signal_right_click = pyqtSignal(int, int)
    signal_shape_moved = pyqtSignal()

    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.drawing_rect_item = False
        self.activate = False
        self.x = 0
        self.y = 0

    # Inherited E-------------------------------------------------------------------------------------------------------
    def mouseMoveEvent(self, event):
        x2 = int(event.scenePos().x())
        y2 = int(event.scenePos().y())
        self.signal_cursor_position.emit(x2, y2)

        if self.drawing_rect_item:
            x = min(self.x, x2)
            y = min(self.y, y2)
            w = abs(x2 - self.x)
            h = abs(y2 - self.y)

            self.signal_draw_temp_roi.emit(x, y, w, h)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        x = int(event.scenePos().x())
        y = int(event.scenePos().y())
        self.signal_cursor_position.emit(x, y)

        if self.activate:
            self.x = x
            self.y = y

            if event.button() == Qt.LeftButton:
                self.signal_draw_new_lane.emit(self.x, self.y)
                self.signal_draw_new_point.emit(self.x, self.y)
                self.drawing_rect_item = True
            elif event.button() == Qt.RightButton:
                self.signal_right_click.emit(self.x, self.y)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.activate:
            x2 = int(event.scenePos().x())
            y2 = int(event.scenePos().y())
            x = min(self.x, x2)
            y = min(self.y, y2)
            w = abs(x2-self.x)
            h = abs(y2-self.y)

            self.drawing_rect_item = False
            self.signal_draw_new_roi.emit(x, y, w, h)
        else:
            self.signal_shape_moved.emit()
            super().mouseReleaseEvent(event)

    # Slots ------------------------------------------------------------------------------------------------------------
    def on_active_draw(self):
        self.activate = not self.activate
