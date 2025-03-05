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
    Description:
        Script for reading radar point clouds, pre-process and display radar data from nuscenes.
"""
import os
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
import pyqtgraph.opengl as gl
from PyQt5.uic.properties import QtGui

import global_config
from PIL import Image
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from data.converters.nuScenes.playground.utils.RadarPointCloud import RadarPointCloud
CFG = global_config.cfg

# Set nuscenes dataset object
nusc = NuScenes(version='v1.0-mini', dataroot='D:/Datasets/NUSCENES/', verbose=True)

# Get the first scene
scene = nusc.scene[0]

# Get first sample of the scene
# This will return tokens to all records for this particular sample
# 6 x Camera, 5 x Radar, 1 x LiDAR
first_sample = nusc.get('sample', scene['first_sample_token'])

# Get image and radar stream
image_tk = first_sample['data']['CAM_FRONT']
radar_tk = first_sample['data']['RADAR_FRONT']

# Get image assigned to image_token
image_data = nusc.get('sample_data', image_tk)

# Get radar cloud assigned to radar token
radar_data = nusc.get('sample_data', radar_tk)

# Get pointcloud
pcl_path = os.path.join('D:/Datasets/NUSCENES/', radar_data['filename'])
pc = RadarPointCloud.from_file(pcl_path)

# Get image
image_path = os.path.join('D:/Datasets/NUSCENES/', image_data['filename'])
img = Image.open(image_path)

# Transform radar pointcloud points in order to project on image.
# First step: transform point cloud to the ego-vehicle radar.
cs_record = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
pc.translate(np.array(cs_record['translation']))

# Second step: transform to the global frame
poserecord = nusc.get('ego_pose', radar_data['ego_pose_token'])
pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
pc.translate(np.array(poserecord['translation']))

# Third step: transform into ego-vehicle stats for image.
poserecord = nusc.get('ego_pose', image_data['ego_pose_token'])
pc.translate(-np.array(poserecord['translation']))
pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

# Fourth step: transofrm into the camera
cs_record = nusc.get('calibrated_sensor', image_data['calibrated_sensor_token'])
pc.translate(-np.array(cs_record['translation']))
pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

# Get depths
#depths = pc.points[2, :]
#coloring = depths

# Get cloud points
#points = pc.points[:3, :]
points = pc.get_points_array()
depths = pc.get_depths_array()
coloring = pc.get_depths_array()

'''
# ========================================================
# OPEN GL
qt_area = QWidget()
gl_view = gl.GLViewWidget()
z_grid = gl.GLGridItem()
z_grid.scale(10, 10, 10)
axis = gl.GLAxisItem(size=QtGui.QVector3D(10, 10, 10), glOptions='opaque')
gl_view.addItem(axis)
pcl = gl.GLScatterPlotItem(pos=points, color=(0, 1, 0, .3), size=0.5, pxMode=False)
gl_view.addItem(pcl)
layout = QGridLayout()
qt_area.setLayout(layout)
qt_area.layout().addWidget(gl_view)
# ========================================================
'''
# Get camera intrinsic parameters
cam_view = np.array(cs_record['camera_intrinsic'])

# Process points and plot on view
view_pad = np.eye(4)
view_pad[:cam_view.shape[0], :cam_view.shape[1]] = cam_view
nr_points = points.shape[1]

points = np.concatenate((points, np.ones((1, nr_points))))
points = np.dot(view_pad, points)
points = points[:3, :]

# Normalize points
points = points / points[2:3, :].repeat(3, 0).reshape(3, nr_points)

# Remove points that are outside bounds.
# Select points that are at least (min_distance) in front of the camera.
mask = np.ones(depths.shape[0], dtype=bool)
mask = np.logical_and(mask, depths > CFG.BASE.MIN_DISTANCE)
mask = np.logical_and(mask, points[0, :] > 1)
mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
mask = np.logical_and(mask, points[1, :] > 1)
mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
points = points[:, mask]
coloring = coloring[mask]


# Set custom point color and coordinates to see if it fits real measurements (in meters)
print('Point Coordinates: \t {}'.format(depths[15]))
print('Point Color: \t {}'.format(coloring[15]))
coloring[15] = 200
# Set plt display
plt.figure(figsize=(9, 16))
plt.imshow(img)
plt.scatter(points[0, :], points[1, :], c=coloring, s=20)
plt.axis('off')
plt.show()
# Render cloud in image
