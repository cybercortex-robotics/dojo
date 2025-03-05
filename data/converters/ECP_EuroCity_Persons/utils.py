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
    EuroCity Persons dataset utils
"""


from data.converters.CyC_DatabaseFormat import create_obj_cls_file


classes = [
    {'name': 'person-group-far-away', 'countable': False, 'id': 0},
    {'name': 'rider', 'countable': True, 'id': 1},
    {'name': 'co-rider', 'countable': True, 'id': 2},
    {'name': 'bicycle', 'countable': True, 'id': 3},
    {'name': 'bicycle-group', 'countable': False, 'id': 4},
    {'name': 'pedestrian', 'countable': True, 'id': 5},
    {'name': 'buggy-group', 'countable': False, 'id': 6},
    {'name': 'motorbike', 'countable': True, 'id': 7},
    {'name': 'scooter', 'countable': True, 'id': 8},
    {'name': 'scooter-group', 'countable': False, 'id': 9},
    {'name': 'buggy', 'countable': True, 'id': 10},
    {'name': 'motorbike-group', 'countable': False, 'id': 11},
    {'name': 'tricycle', 'countable': True, 'id': 12},
    {'name': 'rider+vehicle-group-far-away', 'countable': False, 'id': 13},
    {'name': 'tricycle-group', 'countable': True, 'id': 14},
    {'name': 'wheelchair', 'countable': True, 'id': 15},
    {'name': 'wheelchair-group', 'countable': False, 'id': 16}
]


def identity2cls(identity):
    cls_id = -1

    for cls in classes:
        if cls['name'] == identity:
            cls_id = cls['id']
            break

    if cls_id == -1:
        print('Unknown identity name: {}'.format(identity))
    return cls_id


def create_obj_cls():
    obj_cls = []

    for cls in classes:
        obj_cls.append({
            'name': cls['name'].replace('+', '-'),
            'countable': cls['countable']
        })

    return create_obj_cls_file(obj_cls)


# Method used to filter data
def check_data(bbox):
    area_threshold = 2500  # Set to -1 to deactivate

    # Check if class exists
    if bbox['cls'] == -1:
        return False

    # Check if bbox is in list of identities (filter by class)
    allowed = [identity2cls('pedestrian'), identity2cls('rider')]
    if bbox['cls'] not in allowed:
        return False

    # Check if area it's bigger than a set threshold
    if area_threshold != -1 and area_threshold >= bbox['w'] * bbox['h']:
        return False

    return True


def read_bboxes(parent):
    items = []
    item = {
        'x': parent['x0'],
        'y': parent['y0'],
        'w': parent['x1'] - parent['x0'],
        'h': parent['y1'] - parent['y0'],
        'cls': identity2cls(parent['identity']),
    }

    children = []
    for child in parent['children']:
        children = read_bboxes(child)

    items.append(item)
    for child in children:
        items.append(child)

    return items
