"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

"""Point cloud point ojbect class."""
import numpy as np
from enum import Enum


class Color(Enum):
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)


class Point(object):
    def __init__(self, x, y, z):
        """
            Initialize method for point object.
        :param coordinates: np.array: 3,1 > Array containing x,y,z values.
        :return: None
        """
        self.x = x
        self.y = y
        self.z = z
        self.depth = z
        self.color = Color.WHITE

    def __repr__(self):
        repr_str = 'x: {:.2f}, y: {:.2f}, z: {:.2f}, depth: {:.2f}'
        return repr_str.format(self.get_x, self.get_y, self.get_z, self.get_depth)

    def set_color(self, color_name: str):
        """Set specific color for point."""
        def select_color(col_name) -> Color:
            colors = {
                'white': Color.WHITE,
                'red': Color.RED,
                'green': Color.GREEN,
                'blue': Color.BLUE,
                'black': Color.BLACK
            }
            return colors.get(col_name.lower())
        self.color = select_color(color_name)

    def set_depth(self, depth: float):
        """Sets depth for the current point."""
        self.depth = depth

    def translate(self, x: np.array) -> None:
        """Translate a point by x.
        :param x: <np.array(3,1)>. Translation point.
        """
        self.x += x[0]
        self.y += x[1]
        self.z += x[2]

    def rotate(self, rot_matrix: np.array) -> None:
        """Apply a rotation to the point.
        :param rot_matrix: <np.float: 3,3 >. Rotation matrix."""
        xyz = np.array([self.x, self.y, self.z])
        xyz = np.dot(rot_matrix, xyz)
        self.x, self.y, self.z = xyz

    @property
    def get_x(self):
        return self.x

    @property
    def get_y(self):
        return self.y

    @property
    def get_z(self):
        return self.z

    @property
    def get_depth(self):
        return self.depth

    @property
    def to_list(self):
        return [self.x, self.y, self.z]
