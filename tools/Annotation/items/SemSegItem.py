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


class SemSegItem(QGraphicsPolygonItem, DojoGraphicItem):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, shape_id, class_, brush, pen, *args):
        super().__init__(*args)

        self.pen = pen
        self.item_id = shape_id
        self.class_ = class_[0]
        self.brush = brush
        self.points = dict()
        self.instance_id = 0
        self.moved = False

        self.set_tags()
        self.setAcceptHoverEvents(True)
        self.setToolTip("{} : {}".format(class_[0], class_[1]))

    # Methods ----------------------------------------------------------------------------------------------------------
    def get_center(self):
        return self.boundingRect().center()

    def set_class(self, class_, color):
        self.class_ = class_
        self.brush = color

    def get_class(self):
        return self.class_

    def set_points(self, points):
        self.setPolygon(QPolygonF([QPointF(p[0], p[1]) for p in points]))
        self.update()

    def get_points(self):
        self.update()
        points_str = ""
        for p in self.polygon():
            points_str += "[{} {}]".format(round(p.x(), 3), round(p.y(), 3))

        return "[{}]".format(points_str)

    def get_coord_points(self):
        self.update()
        points = []
        for p in self.polygon():
            points.append([round(p.x(), 3), round(p.y(), 3)])
        return points

    def move_finished(self):
        self.moved = False
        self.signals.signal_edit_finish.emit()

    # Inherited Events -------------------------------------------------------------------------------------------------
    def paint(self, painter, option, widget=None):
        if self.isSelected():
            self.setZValue(1)

        painter.setBrush(self.brush)
        painter.setPen(QPen(self.brush.color()))
        painter.drawPolygon(self.polygon())

    def itemChange(self, change, val):
        if change == QGraphicsItem.ItemPositionChange and not self.moved:
            self.moved = True
            self.signals.signal_item_changed.emit(self.pos().x(),
                                                  self.pos().y(),
                                                  self.sceneBoundingRect().width(),
                                                  self.sceneBoundingRect().height())
        return val
