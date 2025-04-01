"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing,
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import numpy as np
import argparse
import yaml
from toolkit.sensors.pinhole_camera_sensor_model import PinholeCameraSensorModel
from toolkit.env.coordinate_frames_utils import R2euler

np.set_printoptions(precision=8, suppress=True)

def main():
    parser = argparse.ArgumentParser(
        prog="python imu2body_converter.py",
        description="Given camera coordinates in body frame and the imu coordinates in camera frame, "
                    "this script will convert the imu coordinates to body frame."
    )

    parser.add_argument("-cam", default="C:/data/kalibr/rs_d435_01_result/rs_d435_01-camchain-imucam.yaml")
    parser.add_argument("-model", default="C:/dev/src/CyberCortex.AI/core/etc/calibration/cam/cam_rs_d435_231522070156.cal")
    args = parser.parse_args()

    # CAM model
    model = PinholeCameraSensorModel(args.model)

    # Load the YAML file
    with open(args.cam, 'r') as file:
        data = yaml.safe_load(file)

    # Extract the T_cam_imu transformation matrix
    T_cam_imu = np.array(data['cam0']['T_cam_imu'])

    # Invert the transformation matrix
    T_imu_cam = np.linalg.inv(T_cam_imu)

    # Compute the imu coordinates in body frame
    T_body_imu = model.T_cam2body @ T_imu_cam

    # Extract the rotation matrix from T_body_imu
    euler_body_imu = R2euler(T_body_imu[:3, :3]) * 180.0 / np.pi

    print("T_body_imu:\n", T_body_imu)
    print("Euler angles (roll, pitch, yaw) in body frame:\n", euler_body_imu)
    print("Translation vector in body frame:\n", T_body_imu[:3, 3])

if __name__ == "__main__":
    main()