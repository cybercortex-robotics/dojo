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
import numpy as np
from toolkit.env.coordinate_frames_utils import quaternion2R

class CPose(object):
    def __init__(self, _tx=0., _ty=0., _tz=0., _rx=0., _ry=0., _rz=0., _rw=1., _id=-1):
        self.id = _id
        R = quaternion2R(np.array([_rx, _ry, _rz, _rw]))
        t = np.array([_tx, _ty, _tz])

        # 4x4 homogeneous transformation matrix
        self.m_Transform: np.array = np.eye(4)
        self.m_Transform[:3, :3] = R
        self.m_Transform[:3, 3] = t

def tu_CPose(_argv):
    pose = CPose(1, 1, 1, 0, 0, 0, 1, 55)

if __name__ == '__main__':
    try:
        app.run(tu_CPose)
    except SystemExit:
        pass