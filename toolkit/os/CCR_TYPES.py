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

class CcrDatablockKey(object):
    def __init__(self):
        self.nCoreID: int = -1
        self.nFilterID: int = -1

class CcrPoint(object):
    def __init__(self, _id=-1):
        self.id: int = _id
        self.key = CcrDatablockKey()
        self.pt2d = np.array([0., 0.], dtype=np.float32)
        self.depth: float = 0.
        self.score: float = 0.
        self.descriptor: list = []
        self.angle: float = 0.

class CcrVoxel(object):
    def __init__(self):
        self.id: int = -1
        self.pt3d = np.array([0., 0., 0., 1.], dtype=np.float32)
        self.error: float = 0.

def tu_CcrTypes(_argv):
    key = CcrDatablockKey()
    pt = CcrPoint()
    voxel = CcrVoxel()

if __name__ == '__main__':
    try:
        app.run(tu_CcrTypes)
    except SystemExit:
        pass