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
Script for changing the class from one config to another using a mapping conf.
It handles 2D, 3D and SemSeg filters.

Note: Changing for SemSeg includes only changes from framebased_data_descriptor (only polygons).
      SemSeg images remain unchanged and need to be generated again (or implemented this here).
"""

import argparse
import cv2
import numpy as np
import os
import csv
import io, libconf
import shutil
from distutils.dir_util import copy_tree
from data.converters.CyC_DatabaseFormat import CyC_DataBase
from data.types_CyC_TYPES import CyC_FilterType

CHANGE_LIST = [
    str(CyC_FilterType.CyC_OBJECT_DETECTOR_2D_FILTER_TYPE),
    str(CyC_FilterType.CyC_OBJECT_DETECTOR_3D_FILTER_TYPE),
    str(CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE)
]


def convert(args):
    assert os.path.exists(args.path), "Source database does not exist."

    # Read new object classes
    assert os.path.exists(args.map_path), "Mapping conf does not exist."
    with io.open(args.map_path) as f:
        cls_map = libconf.load(f)
    obj_cls_paths = cls_map.Paths
    cls_map = cls_map.Mapping

    # Check paths
    if not os.path.isfile(obj_cls_paths.New) or not os.path.isfile(obj_cls_paths.Old):
        obj_cls_paths = None

    # Create dict for easier handling
    cls_dict = {}
    for key in cls_map.keys():
        cls_dict[str(cls_map[key].ID)] = str(cls_map[key].New_ID)

    # Copy old database to the new location (if locations are the same, update the source database to the new format)
    if args.path != args.output_path:
        print("Copying database to new location..")
        copy_tree(args.path, args.output_path)
        print("Database copied successfully.")

    db_path = args.output_path

    # Read datablock descriptor
    with open('{}/datablock_descriptor.csv'.format(db_path), 'r') as f:
        db_desc = f.readlines()
    db_desc = [[elem for elem in line.strip().split(',')] for line in db_desc[1:]]

    # Chose datastreams
    ds_list = {}
    for line in db_desc:
        if line[3] in CHANGE_LIST:
            ds_list[line[1]] = line[3]
    if args.datastreams != -1:
        pop_keys = []
        for key in ds_list.keys():
            if key not in args.datastreams.split(','):
                pop_keys.append(key)
        for key in pop_keys:
            ds_list.pop(key)

    if len(ds_list.keys()) == 0:
        print('No datastreams selected/eligible to convert. Exiting..')
        return
    print('Converting the following datastreams: ' + ','.join(ds_list.keys()))

    for ds_key in ds_list.keys():
        ds_path = '{}/datastream_{}'.format(db_path, ds_key)

        if ds_list[ds_key] == str(CyC_FilterType.CyC_OBJECT_DETECTOR_2D_FILTER_TYPE):
            with open('{}/framebased_data_descriptor.csv'.format(ds_path), 'r') as file:
                desc_file = file.readlines()
            desc_header = desc_file[0]
            desc_file = desc_file[1:]

            desc_new = []
            for line in desc_file:
                parts = line.strip().split(',')
                try:
                    parts[2] = cls_dict[parts[2]]
                except KeyError:
                    print('Class with id {} was not found.'.format(parts[2]))
                    print('Exiting..')
                    return
                desc_new.append(','.join(parts) + '\n')

            with open('{}/framebased_data_descriptor.csv'.format(ds_path), 'w') as file:
                file.writelines([desc_header] + desc_new)

        elif ds_list[ds_key] == str(CyC_FilterType.CyC_OBJECT_DETECTOR_3D_FILTER_TYPE):
            with open('{}/framebased_data_descriptor.csv'.format(ds_path), 'r') as file:
                desc_file = file.readlines()
            desc_header = desc_file[0]
            desc_file = desc_file[1:]

            desc_new = []
            for line in desc_file:
                parts = line.strip().split(',')
                try:
                    parts[2] = cls_dict[parts[2]]
                except KeyError:
                    print('Class with id {} was not found.'.format(parts[2]))
                    print('Exiting..')
                    return
                desc_new.append(','.join(parts) + '\n')

            with open('{}/framebased_data_descriptor.csv'.format(ds_path), 'w') as file:
                file.writelines([desc_header] + desc_new)

        elif ds_list[ds_key] == str(CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE):
            # Change framebased data descriptor (TODO take in consideration countables?)
            with open('{}/framebased_data_descriptor.csv'.format(ds_path), 'r') as file:
                desc_file = file.readlines()
            desc_header = desc_file[0]
            desc_file = desc_file[1:]

            desc_new = []
            for line in desc_file:
                parts = line.strip().split(',')
                try:
                    parts[2] = cls_dict[parts[2]]
                except KeyError:
                    print('Class with id {} was not found.'.format(parts[2]))
                    print('Exiting..')
                    return
                desc_new.append(','.join(parts) + '\n')

            with open('{}/framebased_data_descriptor.csv'.format(ds_path), 'w') as file:
                file.writelines([desc_header] + desc_new)

            # Change semantic images
            seg_path = os.path.join(ds_path, 'samples/0/left')
            seg_images = os.listdir(seg_path)
            for idx in range(len(seg_images)):
                seg_img = cv2.imread(os.path.join(seg_path, seg_images[idx]))
                new_img = np.zeros_like(seg_img)
                for key in cls_dict.keys():
                    new_img[seg_img == int(key)] = int(cls_dict[key])
                cv2.imwrite(os.path.join(seg_path, seg_images[idx]), new_img)

            # Change instance images
            pass  # TODO

        else:  # Didn't access any option
            continue

        # Update classes file
        if obj_cls_paths is not None:
            if os.path.isfile('{}/object_classes.conf'.format(ds_path)):
                if os.path.isfile('{}/object_classes_old.conf'.format(ds_path)):
                    os.remove('{}/object_classes_old.conf'.format(ds_path))
                os.rename('{}/object_classes.conf'.format(ds_path),
                          '{}/object_classes_old.conf'.format(ds_path))
            shutil.copyfile(obj_cls_paths.New, '{}/object_classes.conf'.format(ds_path))

        print('Finished converting datastream {}'.format(ds_key))
    print('Done')


def main():
    parser = argparse.ArgumentParser(description='Script for changing classes')

    parser.add_argument('--path', '-p', default=r'C:\data\ECP_converted',
                        help='Path to the CyberCortex.AI database')
    # If both paths are the same, will update the source database to the new classes
    parser.add_argument('--output_path', '-o', default=r'C:\data\ECP_converted',
                        help='Location where the new CyberCortex.AI database will be saved.')

    parser.add_argument('--datastreams', '-ds', default='2',
                        help='List of datastreams to be affected. Set to -1 to affect all.')

    parser.add_argument('--map_path', help='Path to the cls mapping conf.',
                        default=r'C:\data\cls_map.conf')

    args = parser.parse_args()

    convert(args=args)


if __name__ == '__main__':
    main()
