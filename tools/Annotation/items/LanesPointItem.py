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

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class LanesPointItem(QGraphicsEllipseItem, DojoGraphicItem):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, point_id, x, y, pen, brush, *args):
        super().__init__(*args)

        self.item_id = point_id
        self.pen = pen
        self.brush = brush

        self.setX(x)
        self.setY(y)
        self.setZValue(1)

        self.set_tags()
        self.setFlag(QGraphicsItem.ItemIsMovable, False)

    # Methods ----------------------------------------------------------------------------------------------------------
    def get_pos(self):
        return round(self.pos().x(), 3), round(self.pos().y(), 3)

    # Inherited Methods ------------------------------------------------------------------------------------------------
    def paint(self, painter, option, widget=None):
        if self.isSelected():
            painter.setPen(QPen(QColor(0, 255, 0)))
            painter.setBrush(QBrush(QColor(0, 255, 0)))
        else:
            painter.setPen(self.pen)
            painter.setBrush(self.brush)

        painter.drawEllipse(self.rect())

    def itemChange(self, change, val):
        if change == QGraphicsItem.ItemPositionChange:
            self.signals.signal_item_changed.emit(round(self.pos().x(), 5),
                                                  round(self.pos().y(), 5),
                                                  self.sceneBoundingRect().width(),
                                                  self.sceneBoundingRect().height())
        return val
