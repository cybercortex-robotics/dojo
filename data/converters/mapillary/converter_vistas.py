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
Script for converting Mapillary Vistas Dataset to CyberCortex.AI format

Dataset link:
    https://www.mapillary.com/dataset/vistas
"""

import argparse
import os.path
import time
import cv2
import json
import numpy as np
from PIL import Image
from data.converters.CyC_DatabaseFormat import *

"""
Dataset Info:

# Tree:
    testing
        images
            <>.jpg
    training
        images
            <>.jpg
        v1.2
            instances
                <>.png
            labels
                <>.png
            panoptic
                <>.png
        v2.0
            instances
                <>.png
            labels
                <>.png
            panoptic
                <>.png
            polygons
                <>.json
    validation
        images
            <>.jpg
        v1.2
            instances
                <>.png
            labels
                <>.png
            panoptic
                <>.png
        v2.0
            instances
                <>.png
            labels
                <>.png
            panoptic
                <>.png
            polygons
                <>.json
    config_v1.2.json
    config_v2.0.json
    demo.py
    
# Config json format:
{
  "labels": [
    {
      "color": [
        165, 
        42, 
        42
      ], 
      "instances": true, 
      "readable": "Bird", 
      "name": "animal--bird", 
      "evaluate": true
    }, 
    {
    ...
    }
  ],
  "version": 1.1, 
  "mapping": "public", 
  "folder_structure": "{split}/{content}/{key:.{22}}.{ext}"
}
"""


# Utils ================================================================================================================
def create_obj_cls(labels):
    classes = []

    for label in labels:
        classes.append({
            'name': label['name'].replace(' ', '_').replace('(', '').replace(')', ''),
            'countable': label['instances']
        })

    return create_obj_cls_file(classes)


def name2class(name, labels):
    for label_id, label in enumerate(labels):
        if label["name"] == name:
            return label_id
    return 0


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


# Convert ==============================================================================================================
def convert_data(args):
    # Calculate splits
    if args.type == 'both':
        splits = ['training', 'validation']
    else:
        splits = [args.type]
    out_dim = (args.width, args.height)  # Output images size

    # Create empty database
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='image', filter_id=1, filter_name='CameraFront')
    db.add_stream(filter_type='semseg', filter_id=2, filter_name='SemSegFront', input_sources=[1])
    db.add_stream(filter_type='image', filter_id=3, filter_name='Panoptic', input_sources=[1])
    db.add_stream(filter_type='2D_obj_det', filter_id=4, filter_name='ObjDet2D', input_sources=[1])
    # db.show_packing_info()
    db.create_db()

    # Prepare timestamp
    ts_start = int(time.time())
    ts_stop = ts_start
    sampling_time = 5

    # Read labels
    with open("{}/config_{}.json".format(args.path, args.version)) as config_file:
        config = json.load(config_file)
    labels = config['labels']

    # Prepare and save object classes file
    obj_cls = create_obj_cls_file(labels)
    db.add_custom(filter_id=2, data=obj_cls, name='object_classes.conf')
    db.add_custom(filter_id=4, data=obj_cls, name='object_classes.conf')

    # For each type (training and validation)
    for s_idx, spl in enumerate(splits):
        root = "{}/{}".format(args.path, spl)
        images_dir = '{}/images'.format(root)
        semseg_dir = '{}/{}/instances'.format(root, args.version)
        pan_dir = '{}/{}/panoptic'.format(root, args.version)
        poly_dir = None
        if args.version == 'v1.2':
            json_path = '{}/{}'.format(pan_dir, "panoptic_2018.json")
        else:  # v2.0
            poly_dir = '{}/{}/polygons'.format(root, args.version)
            json_path = '{}/{}'.format(pan_dir, "panoptic_2020.json")

        # Read json
        try:
            with open(json_path, 'r') as json_file:
                conf = json.load(json_file)
        except FileNotFoundError:
            print('Json {} not found.'.format(json_path))
            continue

        # Count images
        total_images = len(conf['images'])
        if total_images > 0:
            print('Found {} images in the {} set.\n'.format(total_images, spl))
        else:
            print('No images found. Exiting..')
            return
        total_images = min(args.limit, total_images) if args.limit > 0 else total_images
        print('Saving {} images from {} set.'.format(total_images, spl))

        # Read valid classes list
        valid_classes = []
        for cat in conf['categories']:
            if cat['isthing'] > 0:
                valid_classes.append(
                    (cat['id'], cat['name'] if args.version == 'v2.0' else cat['supercategory'])
                )
        valid_classes = dict(valid_classes)

        # Receive data
        for idx in range(total_images):
            # Get info from json
            name = conf['images'][idx]['id']
            im_size = (conf['images'][idx]['width'], conf['images'][idx]['height'])

            # Prepare paths
            rgb_path = '{}/{}.jpg'.format(images_dir, name)
            seg_path = '{}/{}.png'.format(semseg_dir, name)
            pan_path = '{}/{}.png'.format(pan_dir, name)
            poly_path = None
            if args.version == 'v2.0':
                poly_path = '{}/{}.json'.format(poly_dir, name)

            # Check if files exist
            if not os.path.isfile(rgb_path):
                continue
            if not os.path.isfile(seg_path):
                continue
            if not os.path.isfile(pan_path):
                continue
            if poly_path is not None and not os.path.isfile(poly_path):
                continue

            # Process rgb images => RGB Images
            rgb_img = cv2.imread(rgb_path)
            rgb_img_r = cv2.resize(rgb_img, out_dim)

            # Process panoptic images => Panoptic Images
            pan_img = cv2.imread(pan_path)
            pan_img_r = cv2.resize(pan_img, out_dim)

            # Process semseg image => Semantic + Instances
            semseg_raw = Image.open(seg_path)  # contains instance and segmentation
            semseg_array = np.array(semseg_raw, dtype=np.uint16)

            semseg_final = np.array(semseg_array / 256, dtype=np.uint8)
            inst_final = np.array(semseg_array % 256, dtype=np.uint8)

            semseg_final_r = cv2.resize(semseg_final, out_dim)
            inst_final_r = cv2.resize(inst_final, out_dim)

            # Process semseg polygons => Annotation SemSeg
            shape_id, cls_seg, instance, points = [], [], [], []
            if args.version == 'v2.0':
                with open(poly_path if poly_path is not None else '', "r") as poly_f:
                    poly_json = json.load(poly_f)

                for obj in poly_json["objects"]:
                    shape_id.append(obj["id"])
                    cls_seg.append(name2class(obj["label"], labels))
                    instance.append(0)  # No instance id
                    points.append(pts2str(obj["polygon"], out_dim, im_size))

            # Process conf json => Detection 2D
            found = False
            det_id = -1
            while not found:  # Find the image in the list by the id
                det_id += 1
                try:
                    if conf['annotations'][det_id]['image_id'] == name:
                        found = True
                except IndexError:
                    print("Detection not found for image {}".format(name))
                    break

            roi_id, cls_det, x, y, width, height = [], [], [], [], [], []
            if found:
                segments = conf['annotations'][det_id]['segments_info']
                conf['annotations'].pop(det_id)  # To make the process faster

                rate_x = out_dim[0] / im_size[0]
                rate_y = out_dim[1] / im_size[1]

                # Get bboxes
                bbox_id = 0
                for segment in segments:
                    # Check if category is valid and get correct id
                    if segment['category_id'] not in valid_classes.keys():  # ['isthing']
                        continue
                    label_num = 0
                    for label_num in range(len(labels)):
                        if valid_classes[segment['category_id']] == labels[label_num]['name']:
                            break

                    roi_id.append(bbox_id)
                    cls_det.append(label_num)
                    x.append(segment['bbox'][0] * rate_x)
                    y.append(segment['bbox'][1] * rate_y)
                    width.append(segment['bbox'][2] * rate_x)
                    height.append(segment['bbox'][3] * rate_y)
                    bbox_id += 1

            # Pack data
            data = {
                'datastream_1': {  # image
                    'name': '{}.jpg'.format(ts_stop),
                    'image': rgb_img_r
                },
                'datastream_2': {  # sem_seg
                    'name': '{}.png'.format(ts_stop),
                    'semantic': semseg_final_r,
                    'instances': inst_final_r,
                    'shape_id': shape_id,
                    'cls': cls_seg,
                    'instance': instance,
                    'points': points
                },
                'datastream_3': {  # image
                    'name': '{}.png'.format(ts_stop),
                    'image': pan_img_r
                },
                'datastream_4': {  # 2D_det
                    'frame_id': s_idx * total_images + idx,
                    'roi_id': roi_id,
                    'cls': cls_det,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                }
            }

            # Add data to database
            db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

            # Advance timestamp
            ts_start = ts_stop
            ts_stop += sampling_time

            # Print progress
            if idx % 2 == 0:
                print('Done {0} out of {1}.'.format(idx, total_images))
    print('Finished process.')


# View =================================================================================================================
def view_data(args):
    # Set up paths
    images = []
    semseg = []
    panoptic = []
    if args.type != 'both':
        root = "{}/{}".format(args.path, args.type)
        images.extend(['{}/images/{}'.format(root, name) for name in os.listdir(root + '/images')])
        semseg.extend(['{}/{}/instances/{}'.format(root, args.version, name)
                       for name in os.listdir('{}/{}/instances'.format(root, args.version))])
        panoptic.extend(['{}/{}/panoptic/{}'.format(root, args.version, name)
                        for name in os.listdir('{}/{}/panoptic'.format(root, args.version))])
    else:  # training or validation
        splits = ['training', 'validation']
        for spl in splits:
            root = "{}/{}".format(args.path, spl)
            images.extend(['{}/images/{}'.format(root, name) for name in os.listdir(root + '/images')])
            semseg.extend(['{}/{}/instances/{}'.format(root, args.version, name)
                           for name in os.listdir('{}/{}/instances'.format(root, args.version))])
            panoptic.extend(['{}/{}/panoptic/{}'.format(root, args.version, name)
                            for name in os.listdir('{}/{}/panoptic'.format(root, args.version))])

    # Count images
    total_images = len(images)
    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    # View data
    for idx in range(total_images):
        # Check if files exist
        if not os.path.isfile(images[idx]):
            continue
        if not os.path.isfile(semseg[idx]):
            continue
        if not os.path.isfile(panoptic[idx]):
            continue

        # Check if files have the same name
        name = images[idx].split('/')[-1].split('.')[0]
        if name not in semseg[idx] or name not in panoptic[idx]:
            print('The images do not match. Name: 0' + name)
            continue

        # Read images
        raw_img = cv2.imread(images[idx])
        pan_img = cv2.imread(panoptic[idx])

        # Process semseg image
        semseg_raw = Image.open(semseg[idx])  # contains instance and segmentation
        semseg_array = np.array(semseg_raw, dtype=np.uint16)

        semseg_final = np.array(semseg_array / 256, dtype=np.uint8)
        inst_final = np.array(semseg_array % 256, dtype=np.uint8)

        semseg_final = cv2.merge((semseg_final, semseg_final, semseg_final))
        inst_final = cv2.merge((inst_final, inst_final, inst_final)) * 100

        # Compute final image
        final_img = cv2.hconcat([raw_img, semseg_final, inst_final, pan_img])
        final_img = cv2.resize(final_img, (1440, 600))

        # Show final image
        cv2.imshow('Images', final_img)
        cv2.setWindowTitle("Images", '{}/{} : {}'.format(idx, total_images, name))
        cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(description='Convert Mapillary Vistas Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default='C:/dev/Databases/Mapillary_vistas',
                        help='Path to the Mapillary Vistas dataset')
    parser.add_argument('--output_path', '-o', default='C:/dev/Databases/Mapillary_vistas/converted',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--version', default='v2.0',
                        help='Specify the version of the db: v1.2 / v2.0')
    parser.add_argument('--type', default='training',
                        help='training / validation / both')
    parser.add_argument('--width', default=1920, type=int,
                        help='Width of images')
    parser.add_argument('--height', default=1080, type=int,
                        help='Height of images')
    parser.add_argument('--limit', default=25, type=int,
                        help='Number of images to be converted. Set to -1 to convert all db.')

    parser.add_argument('--mode', '-m', default='convert', help='view / convert')

    args = parser.parse_args()

    # Check arguments
    if args.version not in ['v1.2', 'v2.0']:
        raise argparse.ArgumentTypeError(
            "Version is invalid. Try: 'v1.2' or 'v2.0'")
    if args.type not in ['training', 'validation', 'both']:
        raise argparse.ArgumentTypeError(
            "Type is invalid. Try from the following: training, validation or both.")

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        raise argparse.ArgumentTypeError(
            'Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(args.mode))


if __name__ == '__main__':
    main()
