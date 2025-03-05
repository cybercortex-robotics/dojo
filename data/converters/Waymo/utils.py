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
import time
import cv2
import glob
import numpy as np
import open3d as o3d
import tensorflow as tf

from data.converters.CyC_DatabaseFormat import *

from waymo_open_dataset.dataset_pb2 import Frame
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import frame_utils, transform_utils


classes = [
    {'name': 'Unknown', 'id': 0, 'waymo_type': label_pb2.Label.Type.TYPE_UNKNOWN},
    {'name': 'Vehicle', 'id': 1, 'waymo_type': label_pb2.Label.Type.TYPE_VEHICLE},
    {'name': 'Pedestrian', 'id': 2, 'waymo_type': label_pb2.Label.Type.TYPE_PEDESTRIAN},
    {'name': 'Cyclist', 'id': 3, 'waymo_type': label_pb2.Label.Type.TYPE_CYCLIST},
    {'name': 'Sign', 'id': 4, 'waymo_type': label_pb2.Label.Type.TYPE_SIGN},
]

LINE_SEGMENTS = [[0, 1], [1, 3], [3, 2], [2, 0],
                 [4, 5], [5, 7], [7, 6], [6, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]


def type2cls(label_type: label_pb2.Label.Type):
    cls_idx = -1
    for cls in classes:
        if cls['waymo_type'] == label_type:
            cls_idx = cls['id']
            break
    return cls_idx


def create_obj_cls():
    classes_list = []

    for cls in classes:
        classes_list.append({
            'name': cls['name'],
            'countable': True if cls['id'] != 0 else False
        })

    return create_obj_cls_file(classes_list)


def generate_empty_calibration():
    calib = {
        'width': 0,  # Width of image
        'height': 0,  # Height of image

        'rx': 0,  # Rotation X
        'ry': 0,  # Rotation Y
        'rz': 0,  # Rotation Z
        'tx': 0,  # Translation X
        'ty': 0,  # Translation Y
        'tz': 0,  # Translation Z

        'ch': 0,  # Channels
        'fx': 0,  # Focal length X
        'fy': 0,  # Focal length Y
        'cx': 0,  # Optical center X
        'cy': 0,  # Optical center Y
        'px': 0,  # Pixel size X
        'py': 0,  # Pixel size Y
        'dist0': 0,  # Distance coef 0
        'dist1': 0,  # Distance coef 1
        'dist2': 0,  # Distance coef 2
        'dist3': 0,  # Distance coef 3
        'dist4': 0,  # Distance coef 4

        'bline': 0,  # Baseline
    }
    return create_calib_file('Waymo custom calib file', calib)


def vec_to_str(vector):
    str_vec = '['
    str_vec += ' '.join(str(elem) for elem in vector)
    str_vec += ']'
    return str_vec


def generate_matrices_calib(extrinsic, intrinsic):
    calib = generate_empty_calibration()
    calib += '\n\nMatrices =\n{\n'
    if extrinsic is not None:
        calib += '    extrinsic = \"{}\";\n'.format(vec_to_str(extrinsic))
    if intrinsic is not None:
        calib += '    intrinsic = \"{}\";\n'.format(vec_to_str(intrinsic))
    calib += '}'

    return calib


# Decode, change color and resize image
def process_image(img_in, size=(-1, -1)):
    img_out = tf.image.decode_png(img_in)
    img_out = cv2.cvtColor(img_out.numpy(), cv2.COLOR_RGB2BGR)

    if size[0] > 0 and size[1] > 0:
        cv2.resize(img_out, size)
    return img_out


# Processes Lidar point cloud from frame into list of points
def process_lidar(frame: Frame):
    def _range_image_to_pcd(ri_index: int = 0):
        points, points_cp = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose,
            ri_index=ri_index)
        return points, points_cp

    parsed_frame = frame_utils.parse_range_image_and_camera_projection(frame)
    range_images, camera_projections, _, range_image_top_pose = parsed_frame
    frame.lasers.sort(key=lambda laser: laser.name)

    points, points_cp = _range_image_to_pcd()
    points_ri2, points_cp_ri2 = _range_image_to_pcd(1)

    return np.concatenate(points + points_ri2, axis=0)


def generate_det2d(labels):
    det_2d_dict = {
        'id': [],
        'cls': [],
        'x': [],
        'y': [],
        'w': [],
        'h': []
    }

    for label_id, label in enumerate(labels):
        det_2d_dict['id'].append(label_id + 1)  # Roi_id starts at 1
        det_2d_dict['cls'].append(type2cls(label.type))
        det_2d_dict['x'].append(int(label.box.center_x - 0.5 * label.box.length))
        det_2d_dict['y'].append(int(label.box.center_y - 0.5 * label.box.width))
        det_2d_dict['w'].append(label.box.length)
        det_2d_dict['h'].append(label.box.width)

    return det_2d_dict


def generate_det3d(labels):
    det_3d_dict = {
        'id': [],
        'cls': [],
        'x': [], 'y': [], 'z': [],
        'w': [], 'h': [], 'l': [],
        'roll': [], 'pitch': [], 'yaw': []
    }
    for label_id, label in enumerate(labels):
        det_3d_dict['id'].append(label_id + 1)  # Roi_id starts at 1
        det_3d_dict['cls'].append(type2cls(label.type))
        det_3d_dict['x'].append(label.box.center_x)
        det_3d_dict['y'].append(label.box.center_y)
        det_3d_dict['z'].append(label.box.center_z)
        det_3d_dict['w'].append(label.box.width)
        det_3d_dict['h'].append(label.box.height)
        det_3d_dict['l'].append(label.box.length)
        det_3d_dict['roll'].append(0)
        det_3d_dict['pitch'].append(0)
        det_3d_dict['yaw'].append(label.box.heading)

    return det_3d_dict


def get_bbox(label: label_pb2.Label) -> np.ndarray:
    width, length = label.box.width, label.box.length
    return np.array([[-0.5 * length, -0.5 * width],
                     [-0.5 * length, 0.5 * width],
                     [0.5 * length, -0.5 * width],
                     [0.5 * length, 0.5 * width]])


def transform_bbox_waymo(label: label_pb2.Label) -> np.ndarray:
    """Transform object's 3D bounding box using Waymo utils"""
    heading = -label.box.heading
    bbox_corners = get_bbox(label)

    mat = transform_utils.get_yaw_rotation(heading)
    rot_mat = mat.numpy()[:2, :2]

    return bbox_corners @ rot_mat


def build_open3d_bbox(box: np.ndarray, label: label_pb2.Label):
    """Create bounding box's points and lines needed for drawing in open3d"""
    x, y, z = label.box.center_x, label.box.center_y, label.box.center_z

    z_bottom = z - label.box.height / 2
    z_top = z + label.box.height / 2

    points = [[0., 0., 0.]] * box.shape[0] * 2
    for idx in range(box.shape[0]):
        x_, y_ = x + box[idx][0], y + box[idx][1]
        points[idx] = [x_, y_, z_bottom]
        points[idx + 4] = [x_, y_, z_top]

    return points


def show_point_cloud(points: np.ndarray, laser_labels: label_pb2.Label) -> None:
    # pylint: disable=no-member (E1101)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=[0, 0, 0])

    pcd.points = o3d.utility.Vector3dVector(points)

    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)

    for label in laser_labels:
        bbox_corners = transform_bbox_waymo(label)
        # bbox_corners = transform_bbox_custom(label)
        bbox_points = build_open3d_bbox(bbox_corners, label)

        colors = [[1, 0, 0] for _ in range(len(LINE_SEGMENTS))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(LINE_SEGMENTS),
        )

        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    vis.run()
