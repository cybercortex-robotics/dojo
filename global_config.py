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
Define global configuration parameters used across the project.

Users can get the global config by calling:
    from config import cfg
"""
import numpy as np
from easydict import EasyDict as edict
import os

__C = edict()

cfg = __C
__C.BASE = edict()
# CyberCortex.AI
try:
    __C.BASE.PATH = os.environ['CyC_DIR']
except:
    print("Global config: CyC_DIR is undefined. Using default path \"C:/dev/src/CyberCortex.AI\".")
    __C.BASE.PATH = r'C:/dev/src/CyberCortex.AI'

# ===================================
# Set CyberCortex.AI parameters
__C.CyC_INFERENCE = edict()
__C.CyC_INFERENCE.TYPES_FILE = __C.BASE.PATH + r'/inference/include/core/CyC_TYPES.h'
__C.CyC_INFERENCE.PIPELINE_FILE = __C.BASE.PATH + r'/inference/etc/pipelines/control/nuscenes_pipeline.conf'
__C.CyC_INFERENCE.OBJECT_CLASSES_PATH = __C.BASE.PATH + r'/dojo/etc/env/object_classes_carla.conf'

# Database configuration parameters
# ===================================
# Set database path
__C.DB = edict()
__C.DB.BASE_PATH = __C.BASE.PATH + r'/dojo/data/fake_dataset/' # default test dataset
