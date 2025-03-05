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

from PyQt5.QtWidgets import *


class SemSegPixmapItem(QGraphicsPixmapItem, DojoGraphicItem):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, pixmap_image, classes=None, mask=None):
        super().__init__(pixmap_image)

        self.setAcceptHoverEvents(True)

        self.mask = mask
        self.object_classes = classes

        if mask is not None:
            self.w_scale = mask.shape[1] / pixmap_image.size().width()
            self.h_scale = mask.shape[0] / pixmap_image.size().height()

    # Inherited Events -------------------------------------------------------------------------------------------------
    def hoverMoveEvent(self, event):
        if self.mask is None:
            return

        x = int(event.pos().x() * self.w_scale)
        y = int(event.pos().y() * self.h_scale)

        class_idx = self.mask[y][x]

        if self.object_classes.get_class_max_index() < class_idx:
            self.setToolTip("Invalid object class file! Class index {}".format(class_idx))
            return None

        class_name = self.object_classes.get_name_by_index(class_idx)
        self.setToolTip("{}: {}".format(class_idx, class_name))
