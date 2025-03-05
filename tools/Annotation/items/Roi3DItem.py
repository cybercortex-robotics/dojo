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

from toolkit.vision.roi3d_utils import Box

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from math import cos, sin, sqrt, pi, degrees
from pyquaternion import Quaternion

import numpy as np


class ModifiedLineSignal(QObject):
    # Signals ----------------------------------------------------------------------------------------------------------
    signal_mouse_moved = pyqtSignal(QPointF, QPointF)
    signal_key_pressed = pyqtSignal(int)
    signal_wheel_rotated = pyqtSignal(int)
    signal_mouse_released = pyqtSignal()
    signal_mouse_pressed = pyqtSignal()
    signal_line_moved = pyqtSignal()


class BoxSignals(QObject):
    # Signals ----------------------------------------------------------------------------------------------------------
    box_enable_edit = pyqtSignal(bool)
    box_is_moved = pyqtSignal(int)


class AxisGraphicLineItem(QGraphicsLineItem, DojoGraphicItem):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, box_id, *args):
        super().__init__(*args)

        self.box_id = box_id
        self.pos = None
        self.signals = ModifiedLineSignal()

        self.set_tags()

    # Methods ----------------------------------------------------------------------------------------------------------
    def get_box_id(self):
        return self.box_id

    def set_connection(self, slot_moved, slot_key_pressed):
        self.signals.signal_mouse_moved.connect(slot_moved)
        self.signals.signal_key_pressed.connect(slot_key_pressed)

    # Inherited Events -------------------------------------------------------------------------------------------------
    def mouseMoveEvent(self, event):
        self.signals.signal_mouse_moved.emit(self.pos, event.pos())
        self.pos = event.pos()
        super().mouseReleaseEvent(event)

    def mousePressEvent(self, event):
        self.pos = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.pos = None
        self.signals.signal_mouse_released.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self.signals.signal_key_pressed.emit(1)
        if event.key() == Qt.Key_Down:
            self.signals.signal_key_pressed.emit(2)
        if event.key() == Qt.Key_Left:
            self.signals.signal_key_pressed.emit(3)
        if event.key() == Qt.Key_Right:
            self.signals.signal_key_pressed.emit(4)
        if event.key() == Qt.Key_Z:
            self.signals.signal_key_pressed.emit(5)
        if event.key() == Qt.Key_X:
            self.signals.signal_key_pressed.emit(6)
        super().keyPressEvent(event)


class ModifiedQGraphicsLineItem(QGraphicsLineItem, DojoGraphicItem):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, box_id, *args):
        super().__init__(*args)

        self.box_id = box_id
        self.pos = None
        self.signals = ModifiedLineSignal()

        self.set_tags()

    # Methods ----------------------------------------------------------------------------------------------------------
    def get_box_id(self):
        return self.box_id

    def set_connection(self, slot_moved, slot_key_pressed, slot_wheel, slot_pressed):
        self.signals.signal_mouse_moved.connect(slot_moved)
        self.signals.signal_key_pressed.connect(slot_key_pressed)
        self.signals.signal_wheel_rotated.connect(slot_wheel)
        self.signals.signal_mouse_pressed.connect(slot_pressed)

    # Inherited Events -------------------------------------------------------------------------------------------------
    def mouseMoveEvent(self, event):
        self.signals.signal_mouse_moved.emit(self.pos, event.pos())
        self.pos = event.pos()
        super().mouseReleaseEvent(event)

    def mousePressEvent(self, event):
        self.pos = event.pos()
        self.signals.signal_mouse_pressed.emit()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.pos = None
        self.signals.signal_mouse_released.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self.signals.signal_key_pressed.emit(1)
        if event.key() == Qt.Key_Down:
            self.signals.signal_key_pressed.emit(2)
        if event.key() == Qt.Key_Left:
            self.signals.signal_key_pressed.emit(3)
        if event.key() == Qt.Key_Right:
            self.signals.signal_key_pressed.emit(4)
        if event.key() == Qt.Key_Z:
            self.signals.signal_key_pressed.emit(5)
        if event.key() == Qt.Key_X:
            self.signals.signal_key_pressed.emit(6)
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        if self.isSelected():
            self.signals.signal_wheel_rotated.emit(event.delta() / 60)
        super().wheelEvent(event)


class Roi3DItem(QGraphicsItem):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, box: Box, pen, cls, assigned_camera, image_size):
        super().__init__()

        self.assigned_camera = assigned_camera
        self.image_size = image_size
        self.is_selected = False
        self.is_new_box = True
        self.is_saved = False
        self.changed_box = None
        self.box = box
        self.lines = []
        self.pen = pen
        self.cls = cls

        # Box edges
        self.line_front_top = None
        self.line_front_right = None
        self.line_front_bottom = None
        self.line_front_left = None
        self.line_lat_top_right = None
        self.line_lat_top_left = None
        self.line_lat_bot_left = None
        self.line_lat_bot_right = None
        self.line_back_top = None
        self.line_back_right = None
        self.line_back_bottom = None
        self.line_back_left = None
        self.line_front_indicator_one = None
        self.line_front_indicator_two = None
        self.line_top_indicator_one = None
        self.line_top_indicator_two = None
        self.line_x_axis = None
        self.line_y_axis = None
        self.line_z_axis = None

        self.signals = BoxSignals()
        self.set_tags()

    # Static Methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def copy(b):
        return Box(frame_index=b.frame_index,
                   roi_index=b.roi_index,
                   center=[b.center[0], b.center[1], b.center[2]],
                   size=[b.wlh[0], b.wlh[1], b.wlh[2]],
                   orientation=[b.orientation.yaw_pitch_roll[2],
                                b.orientation.yaw_pitch_roll[1],
                                b.orientation.yaw_pitch_roll[0]],
                   name=b.name,
                   color=b.color)

    @staticmethod
    def to_quaternion(rotation_angles):
        """
        Takes the roll, pitch, yaw angles in degrees and generates Quaternion parameters
        :param rotation_angles: list<str> containing angle yaw, pitch, roll values.
        :return: Quaternion w angle, and vector X, Y, Z values.
        """
        roll, pitch, yaw = [np.deg2rad(float(angle)) for angle in rotation_angles]
        cy = cos(yaw / 2)
        sy = sin(yaw / 2)
        cp = cos(pitch / 2)
        sp = sin(pitch / 2)
        cr = cos(roll / 2)
        sr = sin(roll / 2)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return [qw, -qx, qy, qz]

    @staticmethod
    def get_mouse_direction(self, start_pos, end_pos):
        dist_x = end_pos.x() - start_pos.x()
        dist_y = end_pos.y() - start_pos.y()

        if abs(dist_x) > abs(dist_y):
            return dist_x
        return dist_y

    # Methods ----------------------------------------------------------------------------------------------------------
    def set_tags(self):
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)

    def set_connection(self, slot, slot_moved):
        self.signals.box_enable_edit.connect(slot)
        self.signals.box_is_moved.connect(slot_moved)

    def get_corners(self):
        box_2d = self.box
        translation = self.assigned_camera.get_translation()
        rotation = self.to_quaternion(self.assigned_camera.get_rotation())
        rotation = Quaternion(rotation).inverse
        box_2d.translate(-translation)
        box_2d.rotate(rotation)
        corners = box_2d.get_rendering_corners(view=self.assigned_camera.get_intrinsic_matrix(), normalize=True)
        return corners

    def update_box(self):
        translation = self.assigned_camera.get_translation()
        rotation = Quaternion(self.to_quaternion(self.assigned_camera.get_rotation()))

        self.box.rotate(rotation)
        self.box.translate(translation)

    def rotate(self, yaw_angle=0.0, pitch_angle=0.0, roll_angle=0.0):
        if not self.is_saved:
            self.update_box()
        else:
            self.is_saved = False

        my_quaternion = Quaternion(self.to_quaternion([roll_angle, pitch_angle, yaw_angle]))
        self.box.orientation = my_quaternion * self.box.orientation
        self.draw_box(self.get_corners())

    def draw_box(self, corners):
        if self.is_new_box:
            self.line_front_top = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                            int(corners.T[0, 0]),
                                                            int(corners.T[0, 1]),
                                                            int(corners.T[1, 0]),
                                                            int(corners.T[1, 1]))
            self.line_front_top.setPen(self.pen)
            self.line_front_top.set_connection(self.on_mouse_rotation_green_axis,
                                               self.key_pressed_slot,
                                               self.rotate,
                                               self.on_click_change_appearance)
            self.lines.append(self.line_front_top)

            self.line_front_right = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                              int(corners.T[1, 0]),
                                                              int(corners.T[1, 1]),
                                                              int(corners.T[2, 0]),
                                                              int(corners.T[2, 1]))
            self.line_front_right.setPen(self.pen)
            self.line_front_right.set_connection(self.on_mouse_rotation_blue_axis,
                                                 self.key_pressed_slot,
                                                 self.rotate,
                                                 self.on_click_change_appearance)
            self.lines.append(self.line_front_right)

            self.line_front_bottom = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                               int(corners.T[2, 0]),
                                                               int(corners.T[2, 1]),
                                                               int(corners.T[3, 0]),
                                                               int(corners.T[3, 1]))
            self.line_front_bottom.setPen(self.pen)
            self.line_front_bottom.set_connection(self.on_mouse_rotation_green_axis,
                                                  self.key_pressed_slot,
                                                  self.rotate,
                                                  self.on_click_change_appearance)
            self.lines.append(self.line_front_bottom)

            self.line_front_left = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                             int(corners.T[3, 0]),
                                                             int(corners.T[3, 1]),
                                                             int(corners.T[0, 0]),
                                                             int(corners.T[0, 1]))
            self.line_front_left.setPen(self.pen)
            self.line_front_left.set_connection(self.on_mouse_rotation_blue_axis,
                                                self.key_pressed_slot,
                                                self.rotate,
                                                self.on_click_change_appearance)
            self.lines.append(self.line_front_left)

            self.line_lat_top_right = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                                int(corners.T[1, 0]),
                                                                int(corners.T[1, 1]),
                                                                int(corners.T[5, 0]),
                                                                int(corners.T[5, 1]))
            self.line_lat_top_right.setPen(self.pen)
            self.line_lat_top_right.set_connection(self.on_mouse_rotation_red_axis,
                                                   self.key_pressed_slot,
                                                   self.rotate,
                                                   self.on_click_change_appearance)
            self.lines.append(self.line_lat_top_right)

            self.line_lat_top_left = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                               int(corners.T[0, 0]),
                                                               int(corners.T[0, 1]),
                                                               int(corners.T[4, 0]),
                                                               int(corners.T[4, 1]))
            self.line_lat_top_left.setPen(self.pen)
            self.line_lat_top_left.set_connection(self.on_mouse_rotation_red_axis,
                                                  self.key_pressed_slot,
                                                  self.rotate,
                                                  self.on_click_change_appearance)
            self.lines.append(self.line_lat_top_left)

            self.line_lat_bot_left = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                               int(corners.T[3, 0]),
                                                               int(corners.T[3, 1]),
                                                               int(corners.T[7, 0]),
                                                               int(corners.T[7, 1]))
            self.line_lat_bot_left.setPen(self.pen)
            self.line_lat_bot_left.set_connection(self.on_mouse_rotation_red_axis,
                                                  self.key_pressed_slot,
                                                  self.rotate,
                                                  self.on_click_change_appearance)
            self.lines.append(self.line_lat_bot_left)

            self.line_lat_bot_right = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                                int(corners.T[2, 0]),
                                                                int(corners.T[2, 1]),
                                                                int(corners.T[6, 0]),
                                                                int(corners.T[6, 1]))
            self.line_lat_bot_right.setPen(self.pen)
            self.line_lat_bot_right.set_connection(self.on_mouse_rotation_red_axis,
                                                   self.key_pressed_slot,
                                                   self.rotate,
                                                   self.on_click_change_appearance)
            self.lines.append(self.line_lat_bot_right)

            self.line_back_top = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                           int(corners.T[4, 0]),
                                                           int(corners.T[4, 1]),
                                                           int(corners.T[5, 0]),
                                                           int(corners.T[5, 1]))
            self.line_back_top.setPen(self.pen)
            self.line_back_top.set_connection(self.on_mouse_rotation_green_axis,
                                              self.key_pressed_slot,
                                              self.rotate,
                                              self.on_click_change_appearance)
            self.lines.append(self.line_back_top)

            self.line_back_right = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                             int(corners.T[5, 0]),
                                                             int(corners.T[5, 1]),
                                                             int(corners.T[6, 0]),
                                                             int(corners.T[6, 1]))
            self.line_back_right.setPen(self.pen)
            self.line_back_right.set_connection(self.on_mouse_rotation_blue_axis,
                                                self.key_pressed_slot,
                                                self.rotate,
                                                self.on_click_change_appearance)
            self.lines.append(self.line_back_right)

            self.line_back_bottom = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                              int(corners.T[6, 0]),
                                                              int(corners.T[6, 1]),
                                                              int(corners.T[7, 0]),
                                                              int(corners.T[7, 1]))
            self.line_back_bottom.setPen(self.pen)
            self.line_back_bottom.set_connection(self.on_mouse_rotation_green_axis,
                                                 self.key_pressed_slot,
                                                 self.rotate,
                                                 self.on_click_change_appearance)
            self.lines.append(self.line_back_bottom)

            self.line_back_left = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                            int(corners.T[7, 0]),
                                                            int(corners.T[7, 1]),
                                                            int(corners.T[4, 0]),
                                                            int(corners.T[4, 1]))
            self.line_back_left.setPen(self.pen)
            self.line_back_left.set_connection(self.on_mouse_rotation_blue_axis,
                                               self.key_pressed_slot,
                                               self.rotate,
                                               self.on_click_change_appearance)
            self.lines.append(self.line_back_left)

            self.line_front_indicator_one = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                                      int(corners.T[0, 0]),
                                                                      int(corners.T[0, 1]),
                                                                      int(corners.T[2, 0]),
                                                                      int(corners.T[2, 1]))
            self.line_front_indicator_one.setPen(self.pen)
            self.line_front_indicator_one.set_connection(self.on_mouse_rotation_none,
                                                         self.key_pressed_slot,
                                                         self.rotate,
                                                         self.on_click_change_appearance)
            self.lines.append(self.line_front_indicator_one)

            self.line_front_indicator_two = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                                      int(corners.T[1, 0]),
                                                                      int(corners.T[1, 1]),
                                                                      int(corners.T[3, 0]),
                                                                      int(corners.T[3, 1]))
            self.line_front_indicator_two.setPen(self.pen)
            self.line_front_indicator_two.set_connection(self.on_mouse_rotation_none,
                                                         self.key_pressed_slot,
                                                         self.rotate,
                                                         self.on_click_change_appearance)
            self.lines.append(self.line_front_indicator_two)

            self.line_top_indicator_one = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                                    int(corners.T[0, 0]),
                                                                    int(corners.T[0, 1]),
                                                                    int(corners.T[5, 0]),
                                                                    int(corners.T[5, 1]))
            self.line_top_indicator_one.setPen(self.pen)
            self.line_top_indicator_one.set_connection(self.on_mouse_rotation_none,
                                                       self.key_pressed_slot,
                                                       self.rotate,
                                                       self.on_click_change_appearance)
            self.lines.append(self.line_top_indicator_one)

            self.line_top_indicator_two = ModifiedQGraphicsLineItem(self.box.roi_index,
                                                                    int(corners.T[1, 0]),
                                                                    int(corners.T[1, 1]),
                                                                    int(corners.T[4, 0]),
                                                                    int(corners.T[4, 1]))
            self.line_top_indicator_two.setPen(self.pen)
            self.line_top_indicator_two.set_connection(self.on_mouse_rotation_none,
                                                       self.key_pressed_slot,
                                                       self.rotate,
                                                       self.on_click_change_appearance)
            self.lines.append(self.line_top_indicator_two)

            self.line_x_axis = AxisGraphicLineItem(self.box.roi_index,
                                                   int(np.mean(corners.T[:, 0])),
                                                   int(np.mean(corners.T[:, 1])),
                                                   int(np.mean(corners.T[0:4, 0])),
                                                   int(np.mean(corners.T[0:4, 1])))

            self.line_x_axis.setPen(QPen(Qt.red, self.pen.width()))
            self.line_x_axis.set_connection(slot_key_pressed=self.key_pressed_slot,
                                            slot_moved=self.on_mouse_change_dimension_width)
            self.lines.append(self.line_x_axis)
            self.line_x_axis.hide()

            self.line_y_axis = AxisGraphicLineItem(self.box.roi_index,
                                                   int(np.mean(corners.T[:, 0])),
                                                   int(np.mean(corners.T[:, 1])),
                                                   int(np.mean(corners.T[[0, 3, 4, 7], 0])),
                                                   int(np.mean(corners.T[[0, 3, 4, 7], 1])))
            self.line_y_axis.setPen(QPen(Qt.green, self.pen.width()))
            self.line_y_axis.set_connection(slot_key_pressed=self.key_pressed_slot,
                                            slot_moved=self.on_mouse_change_dimension_length)
            self.lines.append(self.line_y_axis)
            self.line_y_axis.hide()

            self.line_z_axis = AxisGraphicLineItem(self.box.roi_index,
                                                   int(np.mean(corners.T[:, 0])),
                                                   int(np.mean(corners.T[:, 1])),
                                                   int(np.mean(corners.T[[0, 1, 4, 5], 0])),
                                                   int(np.mean(corners.T[[0, 1, 4, 5], 1])))
            self.line_z_axis.setPen(QPen(Qt.blue, self.pen.width()))
            self.line_z_axis.set_connection(slot_key_pressed=self.key_pressed_slot,
                                            slot_moved=self.on_mouse_change_dimension_height)
            self.lines.append(self.line_z_axis)
            self.line_z_axis.hide()

            self.is_new_box = False
        else:
            self.line_front_top.setLine(int(corners.T[0, 0]),
                                        int(corners.T[0, 1]),
                                        int(corners.T[1, 0]),
                                        int(corners.T[1, 1]))
            self.line_front_right.setLine(int(corners.T[1, 0]),
                                          int(corners.T[1, 1]),
                                          int(corners.T[2, 0]),
                                          int(corners.T[2, 1]))

            self.line_front_bottom.setLine(int(corners.T[2, 0]),
                                           int(corners.T[2, 1]),
                                           int(corners.T[3, 0]),
                                           int(corners.T[3, 1]))
            self.line_front_left.setLine(int(corners.T[3, 0]),
                                         int(corners.T[3, 1]),
                                         int(corners.T[0, 0]),
                                         int(corners.T[0, 1]))

            self.line_lat_top_right.setLine(int(corners.T[1, 0]),
                                            int(corners.T[1, 1]),
                                            int(corners.T[5, 0]),
                                            int(corners.T[5, 1]))
            self.line_lat_top_left.setLine(int(corners.T[0, 0]),
                                           int(corners.T[0, 1]),
                                           int(corners.T[4, 0]),
                                           int(corners.T[4, 1]))

            self.line_lat_bot_left.setLine(int(corners.T[3, 0]),
                                           int(corners.T[3, 1]),
                                           int(corners.T[7, 0]),
                                           int(corners.T[7, 1]))
            self.line_lat_bot_right.setLine(int(corners.T[2, 0]),
                                            int(corners.T[2, 1]),
                                            int(corners.T[6, 0]),
                                            int(corners.T[6, 1]))

            self.line_back_top.setLine(int(corners.T[4, 0]),
                                       int(corners.T[4, 1]),
                                       int(corners.T[5, 0]),
                                       int(corners.T[5, 1]))
            self.line_back_right.setLine(int(corners.T[5, 0]),
                                         int(corners.T[5, 1]),
                                         int(corners.T[6, 0]),
                                         int(corners.T[6, 1]))
            self.line_back_bottom.setLine(int(corners.T[6, 0]),
                                          int(corners.T[6, 1]),
                                          int(corners.T[7, 0]),
                                          int(corners.T[7, 1]))
            self.line_back_left.setLine(int(corners.T[7, 0]),
                                        int(corners.T[7, 1]),
                                        int(corners.T[4, 0]),
                                        int(corners.T[4, 1]))

            self.line_front_indicator_one.setLine(int(corners.T[0, 0]),
                                                  int(corners.T[0, 1]),
                                                  int(corners.T[2, 0]),
                                                  int(corners.T[2, 1]))
            self.line_front_indicator_two.setLine(int(corners.T[1, 0]),
                                                  int(corners.T[1, 1]),
                                                  int(corners.T[3, 0]),
                                                  int(corners.T[3, 1]))

            self.line_top_indicator_one.setLine(int(corners.T[0, 0]),
                                                int(corners.T[0, 1]),
                                                int(corners.T[5, 0]),
                                                int(corners.T[5, 1]))
            self.line_top_indicator_two.setLine(int(corners.T[1, 0]),
                                                int(corners.T[1, 1]),
                                                int(corners.T[4, 0]),
                                                int(corners.T[4, 1]))

            self.line_x_axis.setLine(int(np.mean(corners.T[:, 0])),
                                     int(np.mean(corners.T[:, 1])),
                                     int(np.mean(corners.T[0:4, 0])),
                                     int(np.mean(corners.T[0:4, 1])))
            self.line_y_axis.setLine(int(np.mean(corners.T[:, 0])),
                                     int(np.mean(corners.T[:, 1])),
                                     int(np.mean(corners.T[[0, 3, 4, 7], 0])),
                                     int(np.mean(corners.T[[0, 3, 4, 7], 1])))
            self.line_z_axis.setLine(int(np.mean(corners.T[:, 0])),
                                     int(np.mean(corners.T[:, 1])),
                                     int(np.mean(corners.T[[0, 1, 4, 5], 0])),
                                     int(np.mean(corners.T[[0, 1, 4, 5], 1])))

            self.signals.box_enable_edit.emit(True)
            self.update_color()

    def key_pressed_slot(self, _type):
        self.signals.box_is_moved.emit(_type)

    def get_item_details(self):
        if not self.is_saved:
            self.update_box()

        self.is_saved = True
        return "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(self.box.frame_index,
                                                              self.box.roi_index,
                                                              self.cls,
                                                              self.box.center[0],
                                                              self.box.center[1],
                                                              self.box.center[2],
                                                              self.box.wlh[0],
                                                              self.box.wlh[1],
                                                              self.box.wlh[2],
                                                              self.box.orientation.yaw_pitch_roll[2],
                                                              self.box.orientation.yaw_pitch_roll[1],
                                                              self.box.orientation.yaw_pitch_roll[0])

    def move_box(self, dx=0, dy=0, dz=0):
        if not self.is_saved:
            self.update_box()
        else:
            self.is_saved = False

        self.box.center[0] += dx
        self.box.center[1] += dy
        self.box.center[2] += dz

    def mouse_offset_to_angle_offset(self, dist_x, dist_y, dist_z):
        d_yaw = dist_x * degrees(pi) / (self.image_size[0] / 2)
        d_pitch = dist_y * degrees(pi) / (self.image_size[1] / 2)
        d_roll = dist_z * degrees(pi) / (sqrt(self.image_size[0] ** 2 + self.image_size[1] ** 2) / 2)

        return d_roll, d_pitch, d_yaw

    def mouse_to_angle(self, dist_x):
        return dist_x * degrees(pi) / (self.image_size[0] / 2)

    def update_color(self):
        self.line_front_top.setPen(self.pen)
        self.line_front_right.setPen(self.pen)
        self.line_front_bottom.setPen(self.pen)
        self.line_front_left.setPen(self.pen)
        self.line_lat_bot_left.setPen(self.pen)
        self.line_lat_top_left.setPen(self.pen)
        self.line_lat_bot_right.setPen(self.pen)
        self.line_lat_top_right.setPen(self.pen)
        self.line_back_bottom.setPen(self.pen)
        self.line_back_left.setPen(self.pen)
        self.line_back_top.setPen(self.pen)
        self.line_back_right.setPen(self.pen)
        self.line_front_indicator_one.setPen(self.pen)
        self.line_front_indicator_two.setPen(self.pen)
        self.line_top_indicator_one.setPen(self.pen)
        self.line_top_indicator_two.setPen(self.pen)

        self.line_x_axis.show()
        self.line_y_axis.show()
        self.line_z_axis.show()

    def hide_axis(self):
        self.line_x_axis.hide()
        self.line_y_axis.hide()
        self.line_z_axis.hide()

    def update_dimension(self, width=0.0, length=0.0, height=0.0):
        if not self.is_saved:
            self.update_box()
        else:
            self.is_saved = False

        self.box.wlh[2] += width
        self.box.wlh[2] = abs(self.box.wlh[2])

        self.box.wlh[0] += length
        self.box.wlh[0] = abs(self.box.wlh[0])

        self.box.wlh[1] += height
        self.box.wlh[1] = abs(self.box.wlh[1])

        self.draw_box(self.get_corners())

    # Slots ------------------------------------------------------------------------------------------------------------
    def on_mouse_rotation_none(self):
        pass

    def on_mouse_rotation_blue_axis(self, start_pos, end_pos):
        dist_x = end_pos.x() - start_pos.x()
        dist_y = -(end_pos.y() - start_pos.y())

        if abs(dist_x) > abs(dist_y):
            self.rotate(yaw_angle=self.mouse_to_angle(dist_x))
        else:
            self.rotate(yaw_angle=self.mouse_to_angle(dist_y))

    def on_mouse_rotation_green_axis(self, start_pos, end_pos):
        dist_x = end_pos.x() - start_pos.x()
        dist_y = -(end_pos.y() - start_pos.y())

        if abs(dist_x) > abs(dist_y):
            self.rotate(pitch_angle=self.mouse_to_angle(dist_x))
        else:
            self.rotate(pitch_angle=self.mouse_to_angle(dist_y))

    def on_mouse_rotation_red_axis(self, start_pos, end_pos):
        dist_x = end_pos.x() - start_pos.x()
        dist_y = (end_pos.y() - start_pos.y())

        if abs(dist_x) > abs(dist_y):
            self.rotate(roll_angle=self.mouse_to_angle(dist_x))
        else:
            self.rotate(roll_angle=self.mouse_to_angle(dist_y))

    def on_mouse_change_dimension_width(self, start_pos, end_pos):
        dist_x = end_pos.x() - start_pos.x()
        dist_y = end_pos.y() - start_pos.y()

        if abs(dist_x) > abs(dist_y):
            sign = np.sign(dist_x)
        else:
            sign = np.sign(dist_y)

        dist_z = sign * sqrt(dist_x**2 + dist_y**2)
        self.update_dimension(width=dist_z / 2)

    def on_mouse_change_dimension_length(self, start_pos, end_pos):
        dist_x = end_pos.x() - start_pos.x()
        dist_y = end_pos.y() - start_pos.y()

        if abs(dist_x) > abs(dist_y):
            sign = np.sign(dist_x)
        else:
            sign = np.sign(dist_y)

        dist_z = sign * sqrt(dist_x**2 + dist_y**2)
        self.update_dimension(length=dist_z / 2)

    def on_mouse_change_dimension_height(self, start_pos, end_pos):
        dist_x = end_pos.x() - start_pos.x()
        dist_y = end_pos.y() - start_pos.y()

        if abs(dist_x) > abs(dist_y):
            sign = np.sign(dist_x)
        else:
            sign = np.sign(dist_y)

        dist_z = sign * sqrt(dist_x ** 2 + dist_y ** 2)
        self.update_dimension(height=dist_z / 2)

    def on_click_change_appearance(self):
        self.update_color()
