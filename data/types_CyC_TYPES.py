"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import os
import json
from global_config import cfg as CFG

# Has to point to the CyC_TYPES.h file in the CyberCortex.AI.inference repository
CyC_types_header = os.path.join(os.path.dirname(__file__), CFG.CyC_INFERENCE.TYPES_FILE)

CyC_FILTER_TYPE_ENUM_NAME = "CyC_FILTER_TYPE"
CyC_DATA_TYPE_ENUM_NAME = "CyC_DATA_TYPE"

if not os.path.exists(CFG.BASE.PATH):
    base_path = os.path.abspath(os.path.join(os.path.abspath(""), "../..", "..", ".."))
    print("Warning: Base path found in global_config.py not found. Defaulting to ", base_path)
    CFG.BASE.PATH = base_path

# conf_file = os.path.join(CFG.BASE.PATH, "dojo", "annotation_tool", "config.json")

# if not os.path.exists(conf_file):
conf_file = "config.json"


def get_datatypes_as_dict(CyC_types_path, enum_name):
    found_filter_type_enum = False
    filter_type_enum_lines = list()
    with open(CyC_types_path, "r") as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            if "enum" in line and enum_name in line:
                found_filter_type_enum = True
            if found_filter_type_enum is True:
                if "}" in line and ";" in line:
                    filter_type_enum_lines.append(line.strip("\n"))
                    break
            if found_filter_type_enum is True:
                filter_type_enum_lines.append(line.strip("\n"))

    # Remove useless lines (comments, empty lines, brackets, semicolons
    useful_idx = list()
    block_comment = False
    for idx, l in enumerate(filter_type_enum_lines):
        if "".join(l.split()).startswith("//"):
            continue
        if "".join(l.split()).startswith("/*"):
            block_comment = True
        if block_comment is True and "".join(l.split()).endswith("*/"):
            block_comment = False
        if block_comment is True:
            continue
        if "=" in l:
            useful_idx.append(idx)

    filter_type_enum_lines = [filter_type_enum_lines[i] for i in useful_idx]
    filter_type_enum_lines = [" ".join(l.split()).split(",")[0] for l in filter_type_enum_lines]

    data_dict = dict()
    for l in filter_type_enum_lines:
        try:
            data_dict[l.split("=")[0].strip()] = int(l.split("=")[1].strip())
        except IndexError:
            return dict()

    return data_dict


class CyC_FilterType:
    def __init__(self):
        """
        Instantiation not needed
        """
        raise NotImplementedError


class CyC_DataType:
    def __init__(self):
        """
        Instantiation not needed
        """
        raise NotImplementedError


if os.path.exists(conf_file):
    # print("Info: Found CyC configuration file (JSON) {}".format(
    #     conf_file
    # ))
    with open(conf_file, "r") as conf:
        conf_json = json.load(conf)
    for data_type in conf_json["CyC_DataTypes"]:
        setattr(CyC_DataType, data_type["name"], data_type["value"])

    for filter_type in conf_json["CyC_FilterTypes"]:
        setattr(CyC_FilterType, filter_type["name"], filter_type["value"])

else:
    print("Warning: Did not find configuration json file. Using deprecated CyC_TYPES.H")
    if not os.path.exists(CyC_types_header):
        print("CyC_TYPES.h: {0} not found. Exiting...".format(CyC_types_header))
        exit(-1)

    d = get_datatypes_as_dict(CyC_types_header, CyC_FILTER_TYPE_ENUM_NAME)
    for k in d:
        setattr(CyC_FilterType, k, d[k])

    d2 = get_datatypes_as_dict(CyC_types_header, CyC_DATA_TYPE_ENUM_NAME)
    for k in d2:
        setattr(CyC_DataType, k, d2[k])
