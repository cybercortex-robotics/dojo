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
Utils for BDD100K
"""

import numpy as np
import cv2


# Built using all the classes from Semantic and Instance Segmentation, Panoptic and Det2D
classes = [{'name': 'invalid', 'countable': False}, {'name': 'sky', 'countable': False},
           {'name': 'road', 'countable': False}, {'name': 'sidewalk', 'countable': False},
           {'name': 'building', 'countable': False}, {'name': 'pole', 'countable': False},
           {'name': 'street_light', 'countable': False}, {'name': 'billboard', 'countable': False},
           {'name': 'vegetation', 'countable': False}, {'name': 'traffic_sign_frame', 'countable': False},
           {'name': 'person', 'countable': True}, {'name': 'traffic_sign', 'countable': True},
           {'name': 'traffic_light', 'countable': True}, {'name': 'car', 'countable': True},
           {'name': 'ego_vehicle', 'countable': False}, {'name': 'fence', 'countable': False},
           {'name': 'terrain', 'countable': False}, {'name': 'truck', 'countable': True},
           {'name': 'motorcycle', 'countable': True}, {'name': 'rider', 'countable': True},
           {'name': 'train', 'countable': True}, {'name': 'bridge', 'countable': False},
           {'name': 'guard_rail', 'countable': False}, {'name': 'polegroup', 'countable': False},
           {'name': 'tunnel', 'countable': False}, {'name': 'fire_hydrant', 'countable': False},
           {'name': 'ground', 'countable': False}, {'name': 'banner', 'countable': False},
           {'name': 'parking', 'countable': False}, {'name': 'trash_can', 'countable': False},
           {'name': 'bus', 'countable': True}, {'name': 'traffic_device', 'countable': False},
           {'name': 'wall', 'countable': False}, {'name': 'caravan', 'countable': True},
           {'name': 'traffic_cone', 'countable': False}, {'name': 'trailer', 'countable': True},
           {'name': 'bicycle', 'countable': True}, {'name': 'bus_stop', 'countable': False},
           {'name': 'parking_sign', 'countable': False}, {'name': 'lane_divider', 'countable': False},
           {'name': 'rail_track', 'countable': False}, {'name': 'mail_box', 'countable': False},
           {'name': 'garage', 'countable': False}, {'name': 'pedestrian', 'countable': True},
           {'name': 'other_vehicle', 'countable': True}, {'name': 'other_person', 'countable': True}]


def name2class(name):
    found_idx = 0  # Index of 'invalid' class
    name = name.replace(' ', '_')
    if name in ['dynamic', 'static', 'unlabeled']:
        return found_idx
    for idx, cls in enumerate(classes):
        if name == cls['name']:
            found_idx = idx
            break
    return found_idx


# Generate semseg, inst and instances from polygons
def generate_semseg(poly_list, cls_list, img_size):
    sem_seg_img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    inst_img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

    # Get countable - for instances
    instances = []
    countable = {}
    for cls_idx, cls in enumerate(classes):
        if cls['countable']:
            countable[cls_idx] = 1

    for idx in range(len(poly_list)):
        inst_id = 0
        if cls_list[idx] in countable.keys():
            inst_id = countable[cls_list[idx]]
            countable[cls_list[idx]] = countable[cls_list[idx]] + 1

        instances.append(inst_id)

        contour = np.array([[int(elem[0]), int(elem[1])] for elem in poly_list[idx]], dtype="int32")
        cv2.fillPoly(sem_seg_img, [contour], (cls_list[idx], cls_list[idx], cls_list[idx]))
        cv2.fillPoly(inst_img, [contour], (inst_id, inst_id, inst_id))

    return instances, sem_seg_img, inst_img


# Create string from polygon points
def pts2str(pts, dim=(-1, -1), im_size=(-1, -1)):
    if dim[0] != -1 and dim[1] != -1:
        rate_x = dim[0]/im_size[0]
        rate_y = dim[1]/im_size[1]
        str_poly = "[{}]".format(
            "".join(["[{} {}]".format(x1 * rate_x, x2 * rate_y) for x1, x2 in pts])
        )
    else:
        str_poly = "[{}]".format(
            "".join(["[{} {}]".format(x1, x2) for x1, x2 in pts])
        )
    return str_poly
