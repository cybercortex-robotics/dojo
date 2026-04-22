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
import pyqtgraph.opengl as gl
from OpenGL.GL import *
from PyQt5.QtGui import QVector3D

from tools.WaypointsPlanner.GL.ListItem import AXIS_WIDTH

"""
 * GLObjects.py
 *
 *  Created on: 01.11.2023
 *      Author: Sorin Grigorescu
"""


class GLAxisItemWaypointsPlanner(gl.GLAxisItem):
    def __init__(self,
                 obj_name="Origin",
                 landmark_id=-1,
                 waypoint_id=-1,
                 position=QVector3D(0, 0, 0),
                 orientation=QVector3D(0, 0, 0),
                 size=None,
                 antialias=True,
                 glOptions='translucent',
                 ignore=False):
        self.orientation = orientation
        self.position = position
        self.name = obj_name
        self.landmark_id = landmark_id
        self.waypoint_id = waypoint_id
        self.ignore = ignore

        super().__init__(size, antialias, glOptions)

    def updateLines(self):
        if self.lineplot is None:
            # still initializing
            return

        x, y, z = self.size()

        pos = np.array([
            [self.position[0], self.position[1], self.position[2], self.position[0], self.position[1], self.position[2] + z],
            [self.position[0], self.position[1], self.position[2], self.position[0], self.position[1] + y, self.position[2]],
            [self.position[0], self.position[1], self.position[2], self.position[0] + x, self.position[1], self.position[2]],
        ], dtype=np.float32).reshape((-1, 3))

        color = np.array([
            [0, 0, 1, 0.6],     # z is blue
            [0, 1, 0, 0.6],     # y is green
            [1, 0, 0, 0.6],     # x is red
        ], dtype=np.float32)

        # color both vertices of each line segment
        color = np.hstack((color, color)).reshape((-1, 4))

        self.lineplot.setData(pos=pos, color=color, width=AXIS_WIDTH)
        self.update()
