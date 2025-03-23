"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing,
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import csv
import os
import shutil
import argparse
import pandas as pd
from data.types_CyC_TYPES import CyC_DataType


def collect_relevant_filters(database_dir):
    cams = []
    imus = []

    filename = os.path.join(database_dir, "datablock_descriptor.csv")
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            # vision_core_id,filter_id,name,type,output_data_type,input_sources
            filter_id = int(row[1])
            output_data_type = int(row[4])
            if output_data_type == CyC_DataType.CyC_IMAGE:
                cams.append(filter_id)
            elif output_data_type == CyC_DataType.CyC_IMU:
                imus.append(filter_id)

    if len(cams) == 0 or len(imus) == 0:
        raise RuntimeError("No IMU or Camera filters.")

    return cams, imus


def process_camera_by_id(source_dir, target_dir, filter_id, camera_id):
    print("Processing filter {} to camera {}...".format(filter_id, camera_id))

    samples_dir = r"datastream_{}\samples\0\left".format(filter_id)
    images_src_dir = os.path.join(source_dir, samples_dir)
    images_dst_dir = os.path.join(target_dir, "cam{}".format(camera_id))

    os.makedirs(images_dst_dir)

    cam_data_descriptor = os.path.join(source_dir, "datastream_{}/data_descriptor.csv".format(filter_id))
    cam_data_path = os.path.join(source_dir, "datastream_{}".format(filter_id))

    # Count the number of rows
    with open(cam_data_descriptor, "r") as fsrc:
        df = pd.read_csv(fsrc, delimiter=',')
        row_count = len(df)
        print("Total rows: {}".format(row_count))

    print("Copying files... ")
    with open(cam_data_descriptor, "r") as fsrc:
        reader = csv.reader(fsrc, delimiter=',')
        next(reader)

        next_percentage = 0.1
        i = 0
        for row in reader:
            ts = row[2]
            filename = os.path.join(cam_data_path, row[3])
            _, file_extension = os.path.splitext(filename)

            image_src_filename = os.path.join(images_src_dir, filename)
            image_dst_filename = os.path.join(images_dst_dir, "{}000000{}".format(ts, file_extension))

            shutil.copy(image_src_filename, image_dst_filename)

            percentage = i / row_count
            if percentage > next_percentage:
                print("{:.0f}%...".format(next_percentage * 100))
                next_percentage += 0.1
            i += 1
    print("Done!")


def process_imu_by_id(source_dir, target_dir, filter_id, imu_id):
    print("Processing filter {} to imu {}...".format(filter_id, imu_id))

    imu_data_descriptor = os.path.join(source_dir, "datastream_{}/data_descriptor.csv".format(filter_id))
    imu_dst_filename = os.path.join(target_dir, "imu{}.csv".format(imu_id))
    imu_data_path = os.path.join(source_dir, "datastream_{}".format(filter_id))

    # Count the number of rows
    with open(imu_data_descriptor, "r") as fsrc:
        df = pd.read_csv(fsrc, delimiter=',')
        row_count = len(df)
        print("Total rows: {}".format(row_count))

    print("Writing csv at {}...".format(imu_dst_filename))
    with open(imu_data_descriptor, "r") as fsrc:
        with open(imu_dst_filename, "w+", newline='') as fdst:
            reader = csv.reader(fsrc, delimiter=',')
            writer = csv.writer(fdst, delimiter=',')

            next(reader)
            writer.writerow(["timestamp", "omega_x", "omega_y", "omega_z", "alpha_x", "alpha_y", "alpha_z"])

            next_percentage = 0.1
            i = 0
            for row in reader:
                imu_data_file = os.path.join(imu_data_path, row[2])
                with open(imu_data_file, "r") as fimu:
                    imu_reader = csv.reader(fimu, delimiter=',')
                    next(imu_reader)
                    for imu_row in imu_reader:
                        timestamp = "{}000000".format(imu_row[0])
                        writer.writerow([timestamp] + imu_row[4:7] + imu_row[1:4])

                percentage = i / row_count
                if percentage > next_percentage:
                    print("{:.0f}%...".format(next_percentage * 100))
                    next_percentage += 0.1
                i += 1
    print("Done")


def main():
    parser = argparse.ArgumentParser(
        prog="python kalibr_convertor.py",
        description="Converts CyberCortex.AI datastreams to intermediary format for Kalibr's bagcreator"
    )

    parser.add_argument("-src", default="C:/data/kalibr/rs_d455_01")
    parser.add_argument("-dst", default="C:/data/kalibr/rs_d455_01_kalibr")

    args = parser.parse_args()
    cams, imus = collect_relevant_filters(args.src)
    for i, cam in enumerate(cams):
        process_camera_by_id(args.src, args.dst, cam, i)
    for i, imu in enumerate(imus):
       process_imu_by_id(args.src, args.dst, imu, i)


if __name__ == "__main__":
    main()
