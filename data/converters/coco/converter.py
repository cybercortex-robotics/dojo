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
Script for converting COCO Dataset to CyberCortex.AI format

Dataset link:
    https://cocodataset.org/#download  - (recommended browser: Mozilla)
"""

import argparse
import os
import time
import json
import cv2
import glob
import numpy as np
from data.converters.CyC_DatabaseFormat import *
from data.converters.coco.utils import *
from pycocotools.mask import *

""" Dataset info:

##################### Tree #####################
train2017
  <>.jpg
val2017
  <>.jpg
annotations --- from Train/Val, Panoptic, Stuff Annotations
  captions_train2017.json   - Contains descriptions for the images. 
  captions_val2017.json     -  example: "A dog jumping over the neck of a giraffe while eating 40 cucumbers".
  person_keypoints_train2017.json   - Contains information regarding person keypoints.
  person_keypoints_val2017.json     -  example: nose, left_ear, right_eye, shoulder...
  instances_train2017.json
  instances_val2017.json
  
  panoptic_train2017.json
  panoptic_train2017
    <>.png
  panoptic_val.2017.json
  panoptic_val.2017
    <>.png
    
  stuff_train2017.json
  stuff_train2017
    <>.png
  stuff_val.2017.json
  stuff_val.2017
    <>.png
    
##################### instances_<>.json content #####################
{
  "info": {...}
  "licenses": {...}
  "images": [{
      "license": 3,
      "file_name": "000000391895.jpg",
      "coco_url": "http://images.cocodataset.org/train2017/000000391895.jpg",
      "height": 360,
      "width": 640,
      "date_captured": "2013-11-14 11:18:45",
      "flickr_url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg",
      "id": 391895
    }, {
      ...
    }
    ...
  ]
  "annotations":[{
      "segmentation": [
        [239.97, 260.24, ...],
        [...],
        ...
      ]
      "area": 2765.148,
      "iscrowd": 0,
      "image_id": 558840,
      "bbox": [199.84, 200.46, 77.71, 70.88]
      "category_id": 58,
      "id": 156
    }, {
      ...
    }
    ...
  ]
  "categories":[{
      "supercategory": "person",
      "id": 1,
      "name": "person"
    }, {
      "supercategory": "vehicle",
      "id": 2,
      "name": "bicycle"
    }, {
      ...
    }
    ...
  ]
}

##################### panoptic_<>.json content #####################
{
  "info": {...}
  "licenses": {...}
  "images": [{
      "license": 3,
      "file_name": "000000391895.jpg",
      "coco_url": "http://images.cocodataset.org/train2017/000000391895.jpg",
      "height": 360,
      "width": 640,
      "date_captured": "2013-11-14 11:18:45",
      "flickr_url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg",
      "id": 391895
    }, {
      ...
    }
    ...
  ]
  "annotations":[{
      "segments_info": [{
            "id": 8345037,
            "category_id": 51,
            "iscrowd": 0,
            "bbox": [0, 14, 434, 374]
            "area": 24315
        }, {
          ...
        }
        ...
      ]
      "file_name": "000000000009.png",
      "image_id": 9
    }, {
      ...
    }
    ...
  ]
  "categories":[{
        "supercategory": "person",
        "isthing": 1
        "id": 1,
        "name": "person"
      }, {
        ...
      }
      ...
    ]
}

##################### stuff_<>.json content #####################
{
  "info": {...}
  "licenses": {...}
  "images": [{
      "license": 3,
      "file_name": "000000391895.jpg",
      "coco_url": "http://images.cocodataset.org/train2017/000000391895.jpg",
      "height": 360,
      "width": 640,
      "date_captured": "2013-11-14 11:18:45",
      "flickr_url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg",
      "id": 391895
    }, {
      ...
    }
    ...
  ]
  "annotations":[{
      "segmentation": {
        "counts": <random string, looks encoded in something>,
        "size": [480, 640]
      }
      "area": 84156.0,
      "iscrowd": 0,
      "image_id": 9,
      "bbox": [0.0, 11.0, 640.0, 469.0]
      "category_id": 121,
      "id": 10000000
    }, {
      ...
    }
    ...
  ]
  "categories":[{
        "supercategory": "textile",
        "id": 92,
        "name": "banner"
      }, {
        ...
      }
      ...
    ]
}
"""


def convert_data(args):
    # Prepare paths
    imgs_dir = '{}/{}{}'.format(args.path, args.split, args.version)
    inst_json_path = '{}/annotations/instances_{}{}.json'.format(args.path, args.split, args.version)
    pan_json_path = '{}/annotations/panoptic_{}{}.json'.format(args.path, args.split, args.version)
    stuff_json_path = '{}/annotations/stuff_{}{}.json'.format(args.path, args.split, args.version)

    # Count data
    total_images = len(os.listdir(imgs_dir))

    if total_images > 0:
        print('Found {} images.'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    if args.limit > 0:
        total_images = min(total_images, args.limit)
        print('Converting only {} images.'.format(total_images))

    # Load jsons
    with open(inst_json_path, 'r') as f:
        inst_json = json.load(f)
    with open(pan_json_path, 'r') as f:
        pan_json = json.load(f)
    with open(stuff_json_path, 'r') as f:
        stuff_json = json.load(f)

    # Create empty database
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='image', filter_id=1, filter_name='Images')
    db.add_stream(filter_type='2D_obj_det', filter_id=2, filter_name='Detection2D', input_sources=[1])
    db.add_stream(filter_type='semseg', filter_id=3, filter_name='SemSeg', input_sources=[1])
    # db.show_packing_info()
    db.create_db()

    # Prepare timestamp
    ts_start = int(time.time())
    ts_stop = ts_start
    sampling_time = 5

    # Prepare and save object classes file
    classes, classes_mapper = process_classes(inst_json['categories'], stuff_json['categories'], pan_json['categories'])
    obj_cls = create_obj_cls_file(classes)
    db.add_custom(filter_id=2, data=obj_cls, name='object_classes.conf')
    db.add_custom(filter_id=3, data=obj_cls, name='object_classes.conf')

    # Receive data
    frame_id = 0
    for image in pan_json['images']:
        img_id = image['id']
        height = image['height']
        width = image['width']
        img_name = image['file_name']

        # Compute ratios (target / current)
        ratio_x = args.width / width
        ratio_y = args.height / height

        # Read and process RGB image
        if not os.path.isfile('{}/{}'.format(imgs_dir, img_name)):
            continue
        rgb_img = cv2.imread('{}/{}'.format(imgs_dir, img_name))
        rgb_img = cv2.resize(rgb_img, (args.width, args.height))

        # Find the annotations of the current image
        inst_anno = []
        for elem in inst_json['annotations']:
            if elem['image_id'] == img_id:
                inst_anno.append(elem)
        stuff_anno = []
        for elem in stuff_json['annotations']:
            if elem['image_id'] == img_id:
                stuff_anno.append(elem)
        pan_anno = []
        for elem in pan_json['annotations']:
            if elem['image_id'] == img_id:
                pan_anno = elem['segments_info']
                break  # There is no need to search further

        # Process Panoptic annotation => Detection 2D
        det_roi_id, det_cls, det_x, det_y, det_width, det_height = [], [], [], [], [], []
        for anno_idx, anno in enumerate(pan_anno):
            det_roi_id.append(anno_idx)
            det_cls.append(classes_mapper[anno['category_id']])
            det_x.append(int(anno['bbox'][0] * ratio_x))
            det_y.append(int(anno['bbox'][1] * ratio_y))
            det_width.append(int(anno['bbox'][2] * ratio_x))
            det_height.append(int(anno['bbox'][3] * ratio_y))

        # Semantic Segmentation variables
        seg_shape_id, seg_cls, seg_pts, pts_list = [], [], [], []
        anno_id = 0

        # Process Instances json => Object Semantic Segmentation
        for anno in inst_anno:
            if type(anno['segmentation']) == dict:
                continue
            for seg in anno['segmentation']:
                if len(seg) % 2 != 0 or len(seg) < 6:
                    continue
                seg_shape_id.append(anno_id)
                seg_cls.append(classes_mapper[anno['category_id']])
                pts_list.append([[int(elem[0]*ratio_x), int(elem[1]*ratio_y)]
                                 for elem in np.reshape(seg, (len(seg) // 2, 2))])
                seg_pts.append(pts2str(pts_list[-1]))
                anno_id += 1

        # Generate semseg from Object Semantic Segmentation
        seg_inst, seg_img, inst_img = generate_semseg(pts_list, seg_cls, (args.height, args.width), classes)

        # Process Stuff json => Others Semantic Segmentation
        for anno in stuff_anno:
            rle = anno['segmentation']
            seg_mask = decode([rle])[:, :, 0]
            seg_mask *= classes_mapper[anno['category_id']]
            seg_mask = cv2.merge((seg_mask, seg_mask, seg_mask))
            seg_mask = cv2.resize(seg_mask, (args.width, args.height), interpolation=cv2.INTER_NEAREST)

            # Merge with the previous segmentations
            seg_img[seg_img == 0] = seg_mask[seg_img == 0]

        # Prepare data
        data = {
            'datastream_1': {  # image
                'name': '{}.jpg'.format(ts_stop),
                'image': rgb_img
            },
            'datastream_2': {  # 2d_det
                'frame_id': frame_id,
                'roi_id': det_roi_id,
                'cls': det_cls,
                'x': det_x,
                'y': det_y,
                'width': det_width,
                'height': det_height
            },
            'datastream_3': {  # sem_seg
                'name': '{}.png'.format(ts_stop),
                'semantic': seg_img,
                'instances': inst_img,
                'shape_id': seg_shape_id,
                'cls': seg_cls,
                'instance': seg_inst,
                'points': seg_pts
            }
        }

        # Add data to database
        db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

        # Advance timestamp
        ts_start = ts_stop
        ts_stop += sampling_time

        # Print progress and check if it's done
        frame_id += 1
        if frame_id % 10 == 0:
            print('Done {0} out of {1}.'.format(frame_id, total_images))
        if frame_id == total_images:
            break
    print('\nFinished converting data: {0} out of {1}.\n'.format(frame_id, total_images))


def view_data(args):
    # Prepare paths
    imgs_dir = '{}/{}{}'.format(args.path, args.split, args.version)
    pan_dir = '{}/annotations/panoptic_{}{}'.format(args.path, args.split, args.version)
    stuff_dir = '{}/annotations/stuff_{}{}_pixelmaps'.format(args.path, args.split, args.version)

    # Count data
    total_images = len(os.listdir(imgs_dir))

    if total_images > 0:
        print('Found {} images.\n'.format(total_images))
    else:
        print('No images found. Exiting..')
        return

    if args.limit > 0:
        total_images = min(total_images, args.limit)
        print('Viewing only {} images.'.format(total_images))

    # View Data
    for image in os.listdir(imgs_dir)[:total_images]:
        name = image.split('.')[0]

        # Check paths
        if not os.path.isfile('{}/{}.jpg'.format(imgs_dir, name)):
            continue
        if not os.path.isfile('{}/{}.png'.format(pan_dir, name)):
            continue
        if not os.path.isfile('{}/{}.png'.format(stuff_dir, name)):
            continue

        # Read images
        rgb_img = cv2.imread('{}/{}.jpg'.format(imgs_dir, name))
        pan_img = cv2.imread('{}/{}.png'.format(pan_dir, name))
        stuff_img = cv2.imread('{}/{}.png'.format(stuff_dir, name))

        final_img = cv2.vconcat([rgb_img, pan_img, stuff_img])
        final_img = cv2.resize(final_img, (600, 600))

        cv2.imshow('image', final_img)
        cv2.setWindowTitle("image", 'Image {0}{1} : {2}'.format(args.split, args.version, name))
        cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(description='Convert COCO Dataset to CyberCortex.AI Dataset format')

    parser.add_argument('--path', '-p', default=r'C:\dev\Databases\COCO',
                        help='Path to the COCO dataset')
    parser.add_argument('--output_path', '-o', default=r'D:\val2017_converted',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--width', default=800, type=int,
                        help='New width of the output image.')
    parser.add_argument('--height', default=600, type=int,
                        help='New height of the output image.')
    parser.add_argument('--split', default='val',
                        help='\'train\' or \'val\'')
    parser.add_argument('--version', default=2017,
                        help='Version of the data. Converter tested on 2017.')
    parser.add_argument('--limit', default=-1, type=int,
                        help='Number of images to be converted. Set to -1 to convert all db.')

    parser.add_argument('--mode', '-m', default='convert', help='view / convert')

    args = parser.parse_args()

    # Check arguments
    if args.split not in ['train', 'val']:
        print('Split is not valid. Try: \'train\' or \'val\'.')
        exit()

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode \'{}\' does not exist. Try \'view\' or \'convert\'.'.format(parser.parse_args().mode))


if __name__ == '__main__':
    main()
