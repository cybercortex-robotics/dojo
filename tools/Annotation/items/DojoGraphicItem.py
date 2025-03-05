"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class DojoGraphicItemSignals(QObject):
    # Signals ----------------------------------------------------------------------------------------------------------
    signal_item_changed = pyqtSignal(int, int, int, int)
    signal_edit_finish = pyqtSignal()


class DojoGraphicItem(QGraphicsItem):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, *args):
        super().__init__(*args)

        self.signals = DojoGraphicItemSignals()
        self.item_id = None
        self.pen = None
        self.delta = 10  # For arrow keys

    # Methods ----------------------------------------------------------------------------------------------------------
    def set_tags(self):
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)

    def set_connection(self, slot_item_changed, slot_edit_finish):
        self.signals.signal_item_changed.connect(slot_item_changed)
        self.signals.signal_edit_finish.connect(slot_edit_finish)

    def set_pen(self, pen):
        self.pen = pen

    def set_item_id(self, item_id):
        self.item_id = item_id

    def get_item_id(self):
        return self.item_id

    # Inherited Events -------------------------------------------------------------------------------------------------
    def keyPressEvent(self, event):
        x = self.pos().x()
        y = self.pos().y()

        if event.key() == Qt.Key_Left or event.key() == Qt.Key_4:
            self.setX(x - self.delta)

        if event.key() == Qt.Key_Right or event.key() == Qt.Key_6:
            self.setX(x + self.delta)

        if event.key() == Qt.Key_Up or event.key() == Qt.Key_8:
            self.setY(y - self.delta)

        if event.key() == Qt.Key_Down or event.key() == Qt.Key_2:
            self.setY(y + self.delta)

        self.signals.signal_edit_finish.emit()
        super().keyPressEvent(event)
