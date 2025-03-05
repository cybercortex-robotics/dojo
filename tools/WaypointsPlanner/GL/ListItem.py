"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing,
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from PyQt5.QtGui import QStandardItem

"""
 * GLObjects.py
 *
 *  Created on: 01.11.2023
 *      Author: Sorin Grigorescu
"""

AXIS_WIDTH = 5
LINE_WIDTH = 5
AXIS_LENGTH = 10
NODE_WIDTH = 0.2
DISTANCE = 10


class ListItem(QStandardItem):
    def __init__(self, text):
        super().__init__(text)

        self.setEditable(False)
        self.setDropEnabled(False)

        f = self.font()
        f.setPixelSize(14)
        self.setFont(f)
