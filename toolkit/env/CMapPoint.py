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
from toolkit.os.CCR_TYPES import CcrVoxel

class CMapPoint(object):
    def __init__(self):
        self.m_CreationKF: int = -1
        self.m_Voxel: CcrVoxel = CcrVoxel()
        self.m_fError: float = 0.
        self.m_Normal: np.array = np.array([0., 0., 0.], dtype=np.float32)
        self.m_mVisibility: dict = {}

def tu_CMap(_argv):
    mp = CMapPoint()

if __name__ == '__main__':
    try:
        app.run(tu_CMap)
    except SystemExit:
        pass