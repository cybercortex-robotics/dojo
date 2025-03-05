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
    Cityscapes classes

Source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
"""
classes = [
    {'name': 'unlabeled', 'id': 0, 'category': 'void', 'catId': 0,
     'hasInstances': False, 'ignoreInEval': True, 'color': (0, 0, 0)},
    {'name': 'ego vehicle', 'id': 1, 'category': 'void', 'catId': 0,
     'hasInstances': False, 'ignoreInEval': True, 'color': (0, 0, 0)},
    {'name': 'rectification border', 'id': 2, 'category': 'void', 'catId': 0,
     'hasInstances': False, 'ignoreInEval': True, 'color': (0, 0, 0)},
    {'name': 'out of roi', 'id': 3, 'category': 'void', 'catId': 0,
     'hasInstances': False, 'ignoreInEval': True, 'color': (0, 0, 0)},
    {'name': 'static', 'id': 4, 'category': 'void', 'catId': 0,
     'hasInstances': False, 'ignoreInEval': True, 'color': (0, 0, 0)},
    {'name': 'dynamic', 'id': 5, 'category': 'void', 'catId': 0,
     'hasInstances': False, 'ignoreInEval': True, 'color': (111, 74, 0)},
    {'name': 'ground', 'id': 6, 'category': 'void', 'catId': 0,
     'hasInstances': False, 'ignoreInEval': True, 'color': (81, 0, 81)},
    {'name': 'road', 'id': 7, 'category': 'flat', 'catId': 1,
     'hasInstances': False, 'ignoreInEval': False, 'color': (128, 64, 128)},
    {'name': 'sidewalk', 'id': 8, 'category': 'flat', 'catId': 1,
     'hasInstances': False, 'ignoreInEval': False, 'color': (244, 35, 232)},
    {'name': 'parking', 'id': 9, 'category': 'flat', 'catId': 1,
     'hasInstances': False, 'ignoreInEval': True, 'color': (250, 170, 160)},
    {'name': 'rail track', 'id': 10, 'category': 'flat', 'catId': 1,
     'hasInstances': False, 'ignoreInEval': True, 'color': (230, 150, 140)},
    {'name': 'building', 'id': 11, 'category': 'construction', 'catId': 2,
     'hasInstances': False, 'ignoreInEval': False, 'color': (70, 70, 70)},
    {'name': 'wall', 'id': 12, 'category': 'construction', 'catId': 2,
     'hasInstances': False, 'ignoreInEval': False, 'color': (102, 102, 156)},
    {'name': 'fence', 'id': 13, 'category': 'construction', 'catId': 2,
     'hasInstances': False, 'ignoreInEval': False, 'color': (190, 153, 153)},
    {'name': 'guard rail', 'id': 14, 'category': 'construction', 'catId': 2,
     'hasInstances': False, 'ignoreInEval': True, 'color': (180, 165, 180)},
    {'name': 'bridge', 'id': 15, 'category': 'construction', 'catId': 2,
     'hasInstances': False, 'ignoreInEval': True, 'color': (150, 100, 100)},
    {'name': 'tunnel', 'id': 16, 'category': 'construction', 'catId': 2,
     'hasInstances': False, 'ignoreInEval': True, 'color': (150, 120, 90)},
    {'name': 'pole', 'id': 17, 'category': 'object', 'catId': 3,
     'hasInstances': False, 'ignoreInEval': False, 'color': (153, 153, 153)},
    {'name': 'polegroup', 'id': 18, 'category': 'object', 'catId': 3,
     'hasInstances': False, 'ignoreInEval': True, 'color': (153, 153, 153)},
    {'name': 'traffic light', 'id': 19, 'category': 'object', 'catId': 3,
     'hasInstances': False, 'ignoreInEval': False, 'color': (250, 170, 30)},
    {'name': 'traffic sign', 'id': 20, 'category': 'object', 'catId': 3,
     'hasInstances': False, 'ignoreInEval': False, 'color': (220, 220, 0)},
    {'name': 'vegetation', 'id': 21, 'category': 'nature', 'catId': 4,
     'hasInstances': False, 'ignoreInEval': False, 'color': (107, 142, 35)},
    {'name': 'terrain', 'id': 22, 'category': 'nature', 'catId': 4,
     'hasInstances': False, 'ignoreInEval': False, 'color': (152, 251, 152)},
    {'name': 'sky', 'id': 23, 'category': 'sky', 'catId': 5,
     'hasInstances': False, 'ignoreInEval': False, 'color': (70, 130, 180)},
    {'name': 'person', 'id': 24, 'category': 'human', 'catId': 6,
     'hasInstances': True, 'ignoreInEval': False, 'color': (220, 20, 60)},
    {'name': 'rider', 'id': 25, 'category': 'human', 'catId': 6,
     'hasInstances': True, 'ignoreInEval': False, 'color': (255, 0, 0)},
    {'name': 'car', 'id': 26, 'category': 'vehicle', 'catId': 7,
     'hasInstances': True, 'ignoreInEval': False, 'color': (0, 0, 142)},
    {'name': 'truck', 'id': 27, 'category': 'vehicle', 'catId': 7,
     'hasInstances': True, 'ignoreInEval': False, 'color': (0, 0, 70)},
    {'name': 'bus', 'id': 28, 'category': 'vehicle', 'catId': 7,
     'hasInstances': True, 'ignoreInEval': False, 'color': (0, 60, 100)},
    {'name': 'caravan', 'id': 29, 'category': 'vehicle', 'catId': 7,
     'hasInstances': True, 'ignoreInEval': True, 'color': (0, 0, 90)},
    {'name': 'trailer', 'id': 30, 'category': 'vehicle', 'catId': 7,
     'hasInstances': True, 'ignoreInEval': True, 'color': (0, 0, 110)},
    {'name': 'train', 'id': 31, 'category': 'vehicle', 'catId': 7,
     'hasInstances': True, 'ignoreInEval': False, 'color': (0, 80, 100)},
    {'name': 'motorcycle', 'id': 32, 'category': 'vehicle', 'catId': 7,
     'hasInstances': True, 'ignoreInEval': False, 'color': (0, 0, 230)},
    {'name': 'bicycle', 'id': 33, 'category': 'vehicle', 'catId': 7,
     'hasInstances': True, 'ignoreInEval': False, 'color': (119, 11, 32)},
    {'name': 'license plate', 'id': -1, 'category': 'vehicle', 'catId': 7,
     'hasInstances': False, 'ignoreInEval': True, 'color': (0, 0, 142)}
]


def name2cls(name):
    cls_idx = -1
    for cls in classes:
        if cls['name'] == name:
            cls_idx = cls['id']
            break
    return cls_idx
