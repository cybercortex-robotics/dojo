"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

"""
 * ArchParser_BaseClass.cpp
 *
 *  Created on: 31.01.2022
 *      Author: Sorin Grigorescu
"""

from absl import app, flags
import logging
from abc import ABC, abstractmethod
import os
import io, libconf

# Initialize logger
logger = logging.getLogger(__name__)

class ArchParser_BaseClass(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def parse_backbone(self):
        pass

    @abstractmethod
    def parse_head(self):
        pass
