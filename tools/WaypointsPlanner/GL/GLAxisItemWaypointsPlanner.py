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
                 glOptions='translucent'):
        super().__init__(size, antialias, glOptions)
        self.orientation = orientation
        self.position = position
        self.name = obj_name
        self.landmark_id = landmark_id
        self.waypoint_id = waypoint_id

    def paint(self):
        self.setupGLState()

        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glLineWidth(AXIS_WIDTH)
        glTranslated(self.position[0], self.position[1], self.position[2])
        glRotate(self.orientation[0], 1, 0, 0)
        glRotate(self.orientation[1], 0, 1, 0)
        glRotate(self.orientation[2], 0, 0, 1)

        glBegin(GL_LINES)

        x, y, z = self.size()
        glColor4f(1, 0, 0, .6)  # x is red
        glVertex3f(0, 0, 0)
        glVertex3f(x, 0, 0)

        glColor4f(0, 1, 0, .6)  # y is green
        glVertex3f(0, 0, 0)
        glVertex3f(0, y, 0)

        glColor4f(0, 0, 1, .6)  # z is blue
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, z)
        glEnd()
