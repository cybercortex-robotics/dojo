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
from toolkit.env.CPose import CPose

class CFrame(object):
    def __init__(self):
        self.m_nFrameID = -1
        self.m_nTimestamp = -1
        self.m_bIsKeyFrame = True
        self.m_Absolute_Body_W: CPose = CPose()
        self.m_Absolute_Cam_C: CPose = CPose()
        self.m_Absolute_Imu_I: CPose = CPose()
        self.keypoints: list = []
        self.m_vMapPointsInFrame: list = []

def tu_CFrame(_argv):
    f = CFrame()

if __name__ == '__main__':
    try:
        app.run(tu_CFrame)
    except SystemExit:
        pass