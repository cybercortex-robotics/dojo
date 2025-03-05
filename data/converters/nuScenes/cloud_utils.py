"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import copy
import struct
import numpy as np
from typing import List, Tuple
from pyquaternion import Quaternion

ply_header_lidar = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''

ply_header_radar = '''
ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property int dyn_prop
property int id
property float rcs
property float vx
property float vy
property float vx_comp
property float vy_comp
property int is_quality_valid
property int ambig_state
property int x_rms
property int y_rms
property int invalid_state
property int pdh0
property int vx_rms
property int vy_rms
end_header
'''



def nbr_dims_lidar() -> int:
    """
    Returns the number of dimensions.
    :return: Number of dimensions.
    """
    return 3

def nbr_dims_radar() -> int:
    """
    Returns the number of dimensions.
    :return: Number of dimensions.
    """
    return 18


# load a lidar point cloud
def load_lidar_from_file(file_name: str):
    """
    Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
    :param file_name: Path of the pointcloud file on disk.
    :return: LidarPointCloud instance (x, y, z,).
    """

    assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

    scan = np.fromfile(file_name, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :nbr_dims_lidar()]
    return points

# load a lidar point cloud
def load_label_info(file_name: str):
    """
    Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
    :param file_name: Path of the pointcloud file on disk.
    :return: LidarPointCloud instance (x, y, z,).
    """

    assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

    scan = np.fromfile(file_name, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :nbr_dims_lidar()]
    return points


def load_radar_from_file(file_name: str):
    """
    Loads RADAR data from a Point Cloud Data file. See details below.
    :param file_name: The path of the pointcloud file.
    :param invalid_states: Radar states to be kept. See details below.
    :param dynprop_states: Radar states to be kept. Use [0, 2, 6] for moving objects only. See details below.
    :param ambig_states: Radar states to be kept. See details below.
    To keep all radar returns, set each state filter to range(18).
    :return: <np.float: d, n>. Point cloud matrix with d dimensions and n points.
    Example of the header fields:
    # .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
    SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
    TYPE F F F I I F F F F F I I I I I I I I
    COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    WIDTH 125
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS 125
    DATA binary
    Below some of the fields are explained in more detail:
    x is front, y is left
    vx, vy are the velocities in m/s.
    vx_comp, vy_comp are the velocities in m/s compensated by the ego motion.
    We recommend using the compensated velocities.
    invalid_state: state of Cluster validity state.
    (Invalid states)
    0x01	invalid due to low RCS
    0x02	invalid due to near-field artefact
    0x03	invalid far range cluster because not confirmed in near range
    0x05	reserved
    0x06	invalid cluster due to high mirror probability
    0x07	Invalid cluster because outside sensor field of view
    0x0d	reserved
    0x0e	invalid cluster because it is a harmonics
    (Valid states)
    0x00	valid
    0x04	valid cluster with low RCS
    0x08	valid cluster with azimuth correction due to elevation
    0x09	valid cluster with high child probability
    0x0a	valid cluster with high probability of being a 50 deg artefact
    0x0b	valid cluster but no local maximum
    0x0c	valid cluster with high artefact probability
    0x0f	valid cluster with above 95m in near range
    0x10	valid cluster with high multi-target probability
    0x11	valid cluster with suspicious angle
    dynProp: Dynamic property of cluster to indicate if is moving or not.
    0: moving
    1: stationary
    2: oncoming
    3: stationary candidate
    4: unknown
    5: crossing stationary
    6: crossing moving
    7: stopped
    ambig_state: State of Doppler (radial velocity) ambiguity solution.
    0: invalid
    1: ambiguous
    2: staggered ramp
    3: unambiguous
    4: stationary candidates
    pdh0: False alarm probability of cluster (i.e. probability of being an artefact caused by multipath or similar).
    0: invalid
    1: <25%
    2: 50%
    3: 75%
    4: 90%
    5: 99%
    6: 99.9%
    7: <=100%
    """

    assert file_name.endswith('.pcd'), 'Unsupported filetype {}'.format(file_name)

    meta = []
    with open(file_name, 'rb') as f:
        for line in f:
            line = line.strip().decode('utf-8')
            meta.append(line)
            if line.startswith('DATA'):
                break

        data_binary = f.read()

    # Get the header rows and check if they appear as expected.
    assert meta[0].startswith('#'), 'First line must be comment'
    assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
    sizes = meta[3].split(' ')[1:]
    types = meta[4].split(' ')[1:]
    counts = meta[5].split(' ')[1:]
    width = int(meta[6].split(' ')[1])
    height = int(meta[7].split(' ')[1])
    data = meta[10].split(' ')[1]
    feature_count = len(types)
    assert width > 0
    assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
    assert height == 1, 'Error: height != 0 not supported!'
    assert data == 'binary'

    # Lookup table for how to decode the binaries.
    unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                     'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                     'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
    types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

    # Decode each point.
    offset = 0
    point_count = width
    points = []
    for i in range(point_count):
        point = []
        for p in range(feature_count):
            start_p = offset
            end_p = start_p + int(sizes[p])
            assert end_p < len(data_binary)
            point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
            point.append(point_p)
            offset = end_p
        points.append(point)

    # A NaN in the first point indicates an empty pointcloud.
    point = np.array(points[0])
    if np.any(np.isnan(point)):
        to_return = []
        for i in range (point_count):
            to_return.append(np.zeros(feature_count))
        return to_return

    # Convert to numpy matrix.
    points = np.array(points).transpose()

    return points.T

def write_lidar_ply(fn, verts):
    """
    Write a PLY file. The data is stored as (x, y, z).
    :param verts: 3D point set.
    :return: None.
    """
    verts_num = len(verts)
    with open(fn, 'wb') as f:
        f.write((ply_header_lidar % dict(vert_num=verts_num)).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f')

def write_radar_ply(fn, verts):
    """
    Write a PLY file. The data is stored as (x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state
        x_rms y_rms invalid_state pdh0 vx_rms vy_rms).
    :param verts: 3D point set.
    :return: None.
    """
    verts_num = len(verts)
    with open(fn, 'wb') as f:
        f.write((ply_header_lidar % dict(vert_num=verts_num)).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %f %f %f %f %f %d %d %d %d %d %d %d %d')

