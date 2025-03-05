"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing,
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""
import math

from OpenGL.GLU import gluUnProject, gluProject
from PyQt5.QtCore import pyqtSignal, Qt
from OpenGL.GL import *
import pyqtgraph.opengl as gl
from numpy.linalg import inv

from tools.WaypointsPlanner.GL.GLAxisItemWaypointsPlanner import GLAxisItemWaypointsPlanner
from tools.WaypointsPlanner.GL.GLCubeItemWaypointsPlanner import GLCubeItemWaypointsPlanner
from tools.WaypointsPlanner.WaypointsPlanner import *

"""
 * GLObjects.py
 *
 *  Created on: 01.11.2023
 *      Author: Sorin Grigorescu
"""


class GLViewWidgetWaypointsPlanner(gl.GLViewWidget):
    signal_mouse_press = pyqtSignal(float, float, float, float, float, bool)
    signal_mouse_release = pyqtSignal(float, float, float, float, float, bool)
    signal_mouse_move = pyqtSignal(float, float, float, float, float, bool)

    def __init__(self):
        super().__init__()
        self.enable_move_camera_view = True
        self.items = []
        self.last_selected_item = None
        self.last_selected_ids = [-1, -1]
        self.way_points = {}
        self.view_port = None
        self.model_view_matrix = None
        self.projection_matrix = None

    def clear(self):
        [self.removeItem(item) for item in self.items]
        self.items.clear()
        self.way_points.clear()

    def getItem(self, landmark_id, waypoint_id):
        return self.way_points["{}.{}".format(landmark_id, waypoint_id)]

    def addItem(self, item):
        self.items.append(item)

        if isinstance(item, GLCubeItemWaypointsPlanner):
            self.way_points["{}.{}".format(item.landmark_id, item.waypoint_id)] = item

        super().addItem(item)

    def get_3d_coord(self, x, y, is_press_event=False, is_right_click_pressed=False):
        self.view_port = glGetIntegerv(GL_VIEWPORT)
        self.model_view_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        self.projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)

        if is_right_click_pressed:
            # TODO: check coordinates of the new box
            pos = self.cameraPosition()

            # Focal distance of the camera
            f = 1/math.tan(float(self.opts['fov']))

            intrinsec_camera_matrix = [[f, 0, self.width()/2],
                                       [0, f, self.height()/2],
                                       [0, 0, 1]]

            roll = 90
            pitch = self.opts['elevation']
            yaw = self.opts['azimuth']

            tr = [[math.cos(roll) * math.cos(pitch),
                   math.cos(roll) * math.sin(pitch) * math.sin(yaw) - math.sin(roll) * math.cos(yaw),
                   math.cos(roll) * math.sin(pitch) * math.cos(yaw) + math.sin(roll) * math.sin(yaw),
                   pos[0]],
                  [math.sin(roll) * math.cos(pitch),
                   math.sin(roll) * math.sin(pitch) * math.sin(yaw) - math.cos(roll) * math.cos(yaw),
                   math.sin(roll) * math.sin(pitch) * math.cos(yaw) + math.cos(roll) * math.sin(yaw),
                   pos[1]],
                  [math.sin(pitch),
                   math.cos(pitch) * math.sin(yaw),
                   math.cos(pitch) * math.cos(yaw),
                   pos[2]],
                  [0, 0, 0, 1]]

            inv_intr_mat = inv(intrinsec_camera_matrix)
            camera_coordinates = inv_intr_mat @ [[x], [y], [1]]

            inv_tr_mat = inv(tr)
            camera_coordinates = [[camera_coordinates[0][0]],
                                  [camera_coordinates[1][0]],
                                  [camera_coordinates[2][0]],
                                  [1]]
            world_coordinates = inv_tr_mat @ camera_coordinates
            window_x, window_y, window_z = gluProject(objX=world_coordinates[0][0],
                                                      objY=world_coordinates[1][0],
                                                      objZ=world_coordinates[2][0],
                                                      model=self.model_view_matrix,
                                                      proj=self.projection_matrix,
                                                      view=self.view_port)

            y = self.view_port[3] - y
            z = glReadPixels(x=window_x,
                             y=window_y,
                             width=1,
                             height=1,
                             format=GL_DEPTH_COMPONENT,
                             type=GL_FLOAT)[0][0]

            position_x, position_y, position_z = gluUnProject(winX=x,
                                                              winY=y,
                                                              winZ=z,
                                                              model=self.model_view_matrix,
                                                              proj=self.projection_matrix,
                                                              view=self.view_port)

            return round(position_x, 2), round(position_y, 2), round(position_z, 2), True

        items = list(filter(lambda item: isinstance(item, GLAxisItemWaypointsPlanner) or
                                         isinstance(item, GLCubeItemWaypointsPlanner),
                            self.itemsAt(region=(x-5, y-5, 10, 10))))

        if len(items) > 0 and is_press_event:
            num_landmarks = len([item for item in items if item.waypoint_id == -1])
            scores = [(item.landmark_id * num_landmarks + item.waypoint_id) for item in items]
            idx = np.argmax(scores)

            self.last_selected_item = items[idx]
            self.last_selected_ids = [self.last_selected_item.landmark_id, self.last_selected_item.waypoint_id]

        elif not is_press_event:
            for item in filter(lambda item: isinstance(item, GLAxisItemWaypointsPlanner) or
                                            isinstance(item, GLCubeItemWaypointsPlanner),
                               self.items):
                if (self.last_selected_ids != [-1, -1] and
                        self.last_selected_ids == [item.landmark_id, item.waypoint_id]):
                    self.last_selected_item = item

        if self.last_selected_item is not None:
            window_x, window_y, window_z = gluProject(objX=self.last_selected_item.position.x(),
                                                      objY=self.last_selected_item.position.y(),
                                                      objZ=self.last_selected_item.position.z(),
                                                      model=self.model_view_matrix,
                                                      proj=self.projection_matrix,
                                                      view=self.view_port)

            if (window_x < 0 or window_y < 0 or window_z < 0 or
                    window_x > self.view_port[2] or window_y > self.view_port[3]):
                return 0., 0., 0., False

            y = self.view_port[3] - y
            z = glReadPixels(x=window_x,
                             y=window_y,
                             width=1,
                             height=1,
                             format=GL_DEPTH_COMPONENT,
                             type=GL_FLOAT)[0][0]

            if z == 1.0:  # sometime this fails for no apparent reason?
                return 0., 0., 0., False

            position_x, position_y, position_z = gluUnProject(winX=x,
                                                              winY=y,
                                                              winZ=z,
                                                              model=self.model_view_matrix,
                                                              proj=self.projection_matrix,
                                                              view=self.view_port)

            return round(position_x, 2), round(position_y, 2), round(position_z, 2), True

        return 0., 0., 0., False

    def mousePressEvent(self, event):
        position_x, position_y, position_z, usable = self.get_3d_coord(event.pos().x(), event.pos().y(),
                                                                       True, event.button() == Qt.RightButton)
        self.signal_mouse_press.emit(event.pos().x(), event.pos().y(), position_x, position_y, position_z, usable)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.last_selected_item = None
        self.last_selected_ids = [-1, -1]

        position_x, position_y, position_z, usable = self.get_3d_coord(event.pos().x(), event.pos().y())
        self.signal_mouse_release.emit(event.pos().x(), event.pos().y(), position_x, position_y, position_z, usable)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        position_x, position_y, position_z, usable = self.get_3d_coord(event.pos().x(), event.pos().y(), False, True)
        self.signal_mouse_move.emit(event.pos().x(), event.pos().y(), position_x, position_y, position_z, usable)

        if self.last_selected_item is None:
            super().mouseMoveEvent(event)
