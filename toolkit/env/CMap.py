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

class CMap(object):
    def __init__(self):
        self.m_nMapPointsCounter: int = -1
        self.m_pMapKeyFrames: list = []
        self.m_MapPoints: list = []

def tu_CMap(_argv):
    map = CMap()

if __name__ == '__main__':
    try:
        app.run(tu_CMap)
    except SystemExit:
        pass