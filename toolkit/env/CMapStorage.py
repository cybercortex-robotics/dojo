"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing,
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from absl import app
import numpy as np
import msgpack
import os
from toolkit.os.CCR_TYPES import CcrPoint
from toolkit.env.CPose import CPose
from toolkit.env.CMap import CMap
from toolkit.env.CMapPoint import CMapPoint
from toolkit.env.CFrame import CFrame

class CMapStorage(object):
    def __init__(self):
        pass

    def unpack_pose(self, unpacker):
        id = unpacker.unpack()
        tx = unpacker.unpack()
        ty = unpacker.unpack()
        tz = unpacker.unpack()
        rx = unpacker.unpack()
        ry = unpacker.unpack()
        rz = unpacker.unpack()
        rw = unpacker.unpack()
        return CPose(tx, ty, tz, rx, ry, rz, rw, id)

    def unpack_obs(self, unpacker):
        pt = CcrPoint()
        pt.id = unpacker.unpack()
        pt.key.nCoreID = unpacker.unpack()
        pt.key.nFilterID = unpacker.unpack()
        pt.pt2d[0] = unpacker.unpack()
        pt.pt2d[1] = unpacker.unpack()
        pt.score = unpacker.unpack()
        pt.depth = unpacker.unpack()
        pt.angle = unpacker.unpack()

        len = unpacker.unpack()
        type = unpacker.unpack()
        pt.descriptor = np.zeros((1, len), dtype=np.float32)
        for i in range(len):
            pt.descriptor[0, i] = unpacker.unpack()
        return pt

    def unpack_map_point(self, unpacker):
        mp = CMapPoint()
        mp.m_CreationKF = unpacker.unpack()
        mp.m_Voxel.id = unpacker.unpack()
        mp.m_Voxel.pt3d[0] = unpacker.unpack()
        mp.m_Voxel.pt3d[1] = unpacker.unpack()
        mp.m_Voxel.pt3d[2] = unpacker.unpack()
        mp.m_Voxel.pt3d[3] = unpacker.unpack()
        mp.m_Voxel.error = unpacker.unpack()

        mp.m_Normal[0] = unpacker.unpack()
        mp.m_Normal[1] = unpacker.unpack()
        mp.m_Normal[2] = unpacker.unpack()

        num_visibility = unpacker.unpack()
        for i in range(num_visibility):
            frame_id = unpacker.unpack()
            obs_id = unpacker.unpack()
            mp.m_mVisibility[frame_id] = obs_id
        return mp

    def load(self, filename, map):
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                unpacker = msgpack.Unpacker(f, raw=False)  # raw=False decodes strings into UTF-8
                map.m_nMapPointsCounter = unpacker.unpack()

                # Load frames
                num_frames = unpacker.unpack()
                for k in range(num_frames):
                    frame = CFrame()
                    frame.m_nFrameID = unpacker.unpack()
                    frame.m_nTimestamp = unpacker.unpack()
                    frame.m_bIsKeyFrame = unpacker.unpack()
                    frame.m_Absolute_Body_W = self.unpack_pose(unpacker)
                    frame.m_Absolute_Cam_C = self.unpack_pose(unpacker)
                    frame.m_Absolute_Imu_I = self.unpack_pose(unpacker)

                    num_observations = unpacker.unpack()
                    for j in range(num_observations):
                        pt = self.unpack_obs(unpacker)
                        frame.keypoints.append(pt)

                    num_map_point_in_frame = unpacker.unpack()
                    for j in range(num_map_point_in_frame):
                        id = unpacker.unpack()
                        frame.m_vMapPointsInFrame.append(id)
                    map.m_pMapKeyFrames.append(frame)

                # Load map points
                num_map_points = unpacker.unpack()
                for k in range(num_map_points):
                    mp = self.unpack_map_point(unpacker)
                    map.m_MapPoints.append(mp)

    def clear(self, map):
        map.m_pMapKeyFrames.clear()
        map.m_MapPoints.clear()

def tu_CMapStorage(_argv):
    map = CMap()
    map_storage = CMapStorage()
    map_storage.load("c:/data/icdt_l6.map", map)

if __name__ == '__main__':
    try:
        app.run(tu_CMapStorage)
    except SystemExit:
        pass
