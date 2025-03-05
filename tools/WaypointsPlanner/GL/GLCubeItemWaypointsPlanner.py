"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing,
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
import pyqtgraph.opengl as gl

from tools.WaypointsPlanner.GL.ListItem import LINE_WIDTH

"""
 * GLObjects.py
 *
 *  Created on: 01.11.2023
 *      Author: Sorin Grigorescu
"""


class GLCubeItemWaypointsPlanner(gl.GLBoxItem):
    def __init__(self, box_name="Box", landmark_id=0, waypoint_id=-1,
                 position=QVector3D(0, 0, 0), size=None, color=None, glOptions='translucent'):
        super().__init__(size, color, glOptions)
        self.position = position
        self.name = box_name
        self.landmark_id = landmark_id
        self.waypoint_id = waypoint_id

    def paint(self):
        glLineWidth(LINE_WIDTH)
        glTranslated(self.position[0], self.position[1], self.position[2])
        super().paint()
