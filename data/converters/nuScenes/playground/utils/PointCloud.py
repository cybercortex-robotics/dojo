"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import numpy as np
from data.converters.nuScenes.playground.utils.Point import Point


class PointCloud(ABC):
    """
        Abstract class for manipulating and viewing point clouds.
        Every point cloud (lidar and radar) consists of points where:
        - Dimensions 0, 1, 2 represent x, y, z coordinates.
            These are modified when the point cloud is rotated or translated.
        - All other dimensions are optional. Hence these have to be manually modified if the reference frame changes.
        """

    def __init__(self, points):
        self.points = self.__encapsulate_points(points)

    @staticmethod
    def __encapsulate_points(points: np.array) -> List[Point]:
        """
        Description : Takes the points coordinates and transforms it to Point object.
        @param: points: Point cloud points.
        @:return: List of <Point> objects."""
        x_coords = points[0, :]
        y_coords = points[1, :]
        z_coords = points[2, :]
        points = np.stack([x_coords, y_coords, z_coords], axis=-1)
        points = [Point(p[0], p[1], p[2]) for p in points]
        return points

    def translate(self, tr_point: np.array):
        """Translate pointcloud."""
        for p in self.points:
            p.translate(tr_point)

    def rotate(self, rot_matrix: np.array):
        """Rotate pointcloud by quaternion values."""
        for p in self.points:
            p.rotate(rot_matrix)

    def get_points_array(self) -> np.array:
        """Return pointcloud as numpy array."""
        return np.array(([point.to_list for point in self.points]), dtype=np.float)

    def get_depths_array(self) -> np.array:
        """Return point cloud depths."""
        return np.array(([point.get_depth for point in self.points]), dtype=np.float)

    @property
    def nbr_dims(self):
        return len(self.points)

    @classmethod
    @abstractmethod
    def from_file(cls, file_name: str) -> 'PointCloud':
        """
        Loads point cloud from disk.
        :param file_name: Path of the pointcloud file on disk.
        :return: PointCloud instance.
        """
        pass

    def set_color_in_radius(self, point: Point, radius: float, center_color: str, neighbors_color: str):
        """Given a point, set the color of this point and within an radius, set the color for the rest of the points.
        :param: point: <Point>. Point object.
        :param: radius: <float>. Range for search.
        :param: center_color: str. Color name for the given point, as the center.
        :param: neighbors_color: str. Color name for the rest of the points found within the given radius."""

