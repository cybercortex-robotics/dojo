"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import csv
import os
import re

root = r'D:/Nuscenes2/'

for dir in os.listdir(root):
    if 'scene' in dir:
        with open((root + dir + '/datastream_1_14/framebsed_data_descriptor.csv'), "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                line = "".join(row)
                print(line)
                if (re.search('[\-][\d]+', line)) is not None:
                    print("!!! NEGATIVE VALUES IN FILE" + (root + dir + '/datastream_1_14/framebsed_data_descriptor.csv') + " IN LINE: " + line)
                    exit(-1)
            csv_file.close()