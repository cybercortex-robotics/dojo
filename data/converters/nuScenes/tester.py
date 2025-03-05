"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from nuscenes.scripts.export_kitti import KittiConverter


kitti_db = KittiConverter(nusc_kitti_dir=r'D:\Projects\Data\Nuscenes\\', split='mini_val')
kitti_db.nuscenes_gt_to_kitti()