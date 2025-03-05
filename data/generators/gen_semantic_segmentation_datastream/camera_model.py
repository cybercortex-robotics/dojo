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


# maps label to attribute name and types for Camera
label_attr_map = {
       "image_width": ["image_width", float],
       "image_height": ["image_height", float],
       "fx": ["fx", float],
       "fy": ["fy", float],
       "cx": ["cx", float],
       "cy": ["cy", float],
       "sx": ["sx", float],
       "sy": ["sy", float],
       "d0": ["d0", float],
       "d1": ["d1", float],
       "d2": ["d2", float],
       "d3": ["d3", float],
       "d4": ["d4", float],
       "baseline": ["baseline", float],
}


class Camera(object):
    def __init__(self, calib_file):
        self.cy = None
        self.cx = None
        self.fy = None
        self.fx = None
        self.image_width = None
        self.image_height = None
        self.sx = None
        self.sy = None
        self.d0 = None
        self.d1 = None
        self.d2 = None
        self.d3 = None
        self.d4 = None
        self.baseline = None
        with open(calib_file, 'r') as input_file:
            for line in input_file:
                # check if the line is empty
                if line.strip():
                    row = line.split()
                    label = row[0]
                    if label in label_attr_map.keys():
                        attr = label_attr_map[label][0]
                        self.__dict__[attr] = float(row[2][:-1])


label_attr_map_cam_pose = {
    # translations
    "x": ["x", float],
    "y": ["y", float],
    "z": ["z", float],
    # rotations
    "pitch": ["pitch", float],
    "roll": ["roll", float],
    "yaw": ["yaw", float],
    "w": ["w", float],
}


class CameraPose(object):
    def __init__(self, calib_file):
        # the rotation is in quaternions
        # the translation is in meters
        self.x = None
        self.y = None
        self.z = None
        self.pitch = None
        self.roll = None
        self.yaw = None
        self.w = None

        with open(calib_file, 'r') as input_file:
            for line in input_file:
                # check if the line is empty
                if line.strip():
                    row = line.split()
                    label = row[0]
                    if label in label_attr_map_cam_pose.keys():
                        attr = label_attr_map_cam_pose[label][0]
                        self.__dict__[attr] = float(row[2][:-1])

    def _quaternion_to_euler(self):

        # roll(x - axis rotation)
        sinr_cosp = 2 * (self.w * self.roll + self.pitch * self.yaw)
        cosr_cosp = 1 - 2 * (self.roll * self.roll + self.pitch * self.pitch)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch(y - axis rotation)
        sinp = 2 * (self.w * self.pitch - self.yaw * self.roll)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp) # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # yaw(z - axis rotation)
        siny_cosp = 2 * (self.w * self.yaw + self.roll * self.pitch)
        cosy_cosp = 1 - 2 * (self.pitch * self.pitch + self.yaw * self.yaw)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


class RoadModel(object):
    def __init__(self):
        self.roll = 0.0
        self.pitch = -0.5
        self.yaw = 40.0 * (np.pi / 180.0)
        self.lateral_offset = 0.0
        self.width = 0.5
        self.curvature = 0.0

        # Fixed parameters
        self.cam_height = 1.5
        self.lookahead_offset = 0.0

    def update(self, curvature):
        self.curvature = curvature * 0.25
        # calculate width
        """
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(0, 100);
        self.width = 0.5 - std::abs(m_RoadModel.curvature) * dist(rng) * 0.01F;
        """
        d = np.random.uniform(1, 100, 1)
        self.width = (0.5 - abs(self.curvature) * 0.01 * d) * 10