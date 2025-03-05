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
    Asta inseamna urmatoarea mapare de la nuscenes la noi:
    0, 1, 2, 3, 4 -> 0
    5, 6, 7, 8, 9 -> 1
    13 -> 2
    14, 16, 17 -> 3
    11, 12 -> 4
    15 -> 5
    10 -> 6
"""

from absl import app, flags, logging
import os
import csv
import math
from pathlib import Path
from copy import deepcopy
import json
import pandas as pd
import global_config

CFG = global_config.cfg
dataset_path = r'D:/NuScenes/'

object_detection_datastreams = list(range(14, 26))
nuScenes2Rovis_dict = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 6,
    11: 6,
    12: 4,
    13: 4,
    14: 2,
    15: 3,
    16: 5,
    17: 3,
    18: 3
}

def is_int(s: str) -> bool:
    """
    Verifies if a string can be converted to a int value.
    :param s:
    :return:
    """
    try:
        int(s)
        return True
    except ValueError:
        return  False

def change_class(path:str):
    """
    Changes the NuScenes classes to Rovis classes
    :param path:
    :return:
    """
    framebased_data_descriptor_path = os.path.join(path, "framebased_data_descriptor.csv")

    framebased_data_descriptor = pd.read_csv(framebased_data_descriptor_path)
    framebased_data_descriptor['cls'] = framebased_data_descriptor['cls'].apply(lambda x: nuScenes2Rovis_dict[x])
    framebased_data_descriptor.to_csv(framebased_data_descriptor_path, index=False, sep=',')

def main(_argv):
    for scene in os.listdir(dataset_path):
        scene_path = os.path.join(dataset_path, scene)
        for datastream in os.listdir(scene_path):
            datastream_path = os.path.join(scene_path, datastream)
            datastream_id = datastream_path.split('_')[-1]
            if is_int(datastream_id) and int(datastream_id) in object_detection_datastreams:
                change_class(datastream_path)
    return 0

if __name__ == '__main__':
    app.run(main)