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
Utils for COCO database converter
"""

import numpy as np
import cv2


# Processes classes into obj_classes and a classes mapper coco -> CyberCortex.AI
def process_classes(inst_cat, stuff_cat, pan_cat):
    classes = [{  # Start with invalid class
        'name': 'invalid',  # Index 0
        'countable': False
    }]
    classes_mapper = [0]  # classes_mapper[coco_id] = cyc_id

    # Find max ID for classes
    max_id = 0
    for cat in inst_cat:
        max_id = max(max_id, cat['id'])
    for cat in stuff_cat:
        max_id = max(max_id, cat['id'])
    for cat in pan_cat:
        max_id = max(max_id, cat['id'])

    # Parse categories to find all classes using the indexes
    cyc_cls_idx = 1
    for idx in range(1, max_id+1):
        name = ''
        countable = False
        found = False

        # Search inst_cat (objects)
        for cat in inst_cat:
            if cat['id'] == idx:
                found = True
                name = cat['name']
                countable = True
                break

        # Search stuff_cat (background)
        if not found:
            for cat in stuff_cat:
                if cat['id'] == idx:
                    found = True
                    name = cat['name']
                    countable = False
                    break

        # Search pan_cat (both)
        if not found:
            for cat in pan_cat:
                if cat['id'] == idx:
                    found = True
                    name = cat['name']
                    countable = True if cat['isthing'] == 1 else False
                    break

        if found:
            classes.append({
                'name': name.replace(' ', '_').replace('-', '_'),
                'countable': countable
            })
            classes_mapper.append(cyc_cls_idx)
            cyc_cls_idx += 1
        else:  # not found
            classes_mapper.append(0)

    return classes, classes_mapper


# Generate semseg, inst and instances from polygons
def generate_semseg(poly_list, cls_list, img_size, classes):
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
def pts2str(pts):
    return "[{}]".format(
        "".join(["[{} {}]".format(x1, x2) for x1, x2 in pts])
    )
