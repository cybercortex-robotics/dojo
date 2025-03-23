"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from absl import app
import os
import io, libconf
import numpy as np
from ..env import coordinate_frames_utils as frames

class ImuSensorModel(object):
    def __init__(self, calibration_file):
        assert os.path.exists(calibration_file), 'Path to calibration file invalid.'
        self.rotation = np.zeros((3))
        self.translation = np.zeros((3))
        self.read_calib_file(calibration_file)

    def __str__(self):
        str_display = "Rotation:\t\t{0}\n".format(self.rotation)
        str_display += "Translation:\t{0}\n".format(self.translation)
        return str_display

    def read_calib_file(self, calib_file_path):
        """Read configuration file and store the content in a dict."""
        assert os.path.exists(calib_file_path), 'Path to camera calibration file invalid.'
        with io.open(calib_file_path) as f:
            sections = libconf.load(f)

            # Read extrinsics
            self.rotation[0] = sections['Pose']['Rotation']['x']
            self.rotation[1] = sections['Pose']['Rotation']['y']
            self.rotation[2] = sections['Pose']['Rotation']['z']
            self.translation[0] = sections['Pose']['Translation']['x']
            self.translation[1] = sections['Pose']['Translation']['y']
            self.translation[2] = sections['Pose']['Translation']['z']


def tu_sensor_model(_argv):
    np.set_printoptions(suppress=True)
    # M_veh2cam = frames.transform_mat_XYZ(1.51, 0.015, 1.70, np.deg2rad(0.), np.deg2rad(90.), np.deg2rad(-90.))
    # print(M_veh2cam)

    calibration_file = r"C:\dev\src\CyberCortex.AI\core\etc\calibration\cam\cam_rs_231122300048.cal"
    sensor_model = ImuSensorModel(calibration_file)
    print(sensor_model)

if __name__ == '__main__':
    try:
        app.run(tu_sensor_model)
    except SystemExit:
        pass
