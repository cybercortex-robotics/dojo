"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from tools.Annotation.items.DojoGraphicItem import DojoGraphicItem

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class SemSegPointItem(QGraphicsEllipseItem, DojoGraphicItem):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, item_id, point_index, pen, brush, x, y, *args):
        super().__init__(*args)

        self.setX(x)
        self.setY(y)
        self.setBrush(brush)
        self.setPen(pen)
        self.item_id = item_id
        self.point_index = point_index
        self.set_tags()
        self.setZValue(1)
        self.shape_point = False

    # Methods ----------------------------------------------------------------------------------------------------------
    def set_as_shape_point(self):
        self.shape_point = True

    def change_brush(self, brush):
        self.setBrush(brush)
        self.update()

    def change_pen(self, pen):
        self.pen = pen
        self.setPen(pen)
        self.update()

    def get_index_point(self):
        return self.point_index

    def get_pos(self):
        return self.pos().x(), self.pos().y()

    # Inherited Events -------------------------------------------------------------------------------------------------
    def hoverMoveEvent(self, event):
        if self.shape_point:
            self.setCursor(Qt.OpenHandCursor)
        super().hoverMoveEvent(event)

    def mouseMoveEvent(self, event):
        if self.shape_point:
            self.setCursor(Qt.ClosedHandCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.shape_point:
            self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)
        self.signals.signal_edit_finish.emit()

    def itemChange(self, change, val):
        if change == QGraphicsItem.ItemPositionChange:
            self.signals.signal_item_changed.emit(round(self.pos().x(), 5),
                                                  round(self.pos().y(), 5),
                                                  self.sceneBoundingRect().width(),
                                                  self.sceneBoundingRect().height())
        return val
