"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""


class DatastreamErrorCodes(object):
    OK = 0
    INVALID_BASE_PATH = -1
    BLOCKCHAIN_FILE_NOT_FOUND = -2
    OVERLAY_DATASTREAM_NOT_FOUND = -3
    DATASTREAM_ALREADY_PRESENT = -4
    FILTER_DATASET_PRESENT = -5
