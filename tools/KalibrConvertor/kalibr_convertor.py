import csv
import os
import shutil
import argparse
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

    print("Copying files... ")
    files = os.listdir(images_src_dir)
    next_percentage = 0.1
    for i, image_name in enumerate(files):
        filename, file_extension = os.path.splitext(image_name)
        new_image_name = "{}000000{}".format(filename, file_extension)

        image_src_filename = os.path.join(images_src_dir, image_name)
        image_dst_filename = os.path.join(images_dst_dir, new_image_name)

        shutil.copy(image_src_filename, image_dst_filename)

        percentage = i / len(files)
        if percentage > next_percentage:
            print("{:.0f}%...".format(next_percentage * 100))
            next_percentage += 0.1

    print("Done!")


def process_imu_by_id(source_dir, target_dir, filter_id, imu_id):
    print("Processing filter {} to camera {}...".format(filter_id, imu_id))

    imu_src_filename = os.path.join(source_dir, "datastream_{}/data_descriptor.csv".format(filter_id))
    imu_dst_filename = os.path.join(target_dir, "imu{}.csv".format(imu_id))

    print("Writing csv at {}...".format(imu_dst_filename))
    with open(imu_src_filename, "r") as fsrc:
        with open(imu_dst_filename, "w+", newline='') as fdst:
            reader = csv.reader(fsrc, delimiter=',')
            writer = csv.writer(fdst, delimiter=',')

            next(reader)
            writer.writerow(["timestamp", "omega_x", "omega_y", "omega_z", "alpha_x", "alpha_y", "alpha_z"])

            for row in reader:
                # timestamp_start,timestamp_stop,sampling_time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,yaw,pitch,roll
                timestamp = "{}000000".format(row[1])
                writer.writerow([timestamp] + row[3:9])

    print("Done")


def main():
    parser = argparse.ArgumentParser(
        prog="python kalibr_convertor.py",
        description="Converts CyberCortex.AI datastreams to intermediary format for Kalibr's bagcreator"
    )

    parser.add_argument("-src", required=True)
    parser.add_argument("-dst", required=True)

    args = parser.parse_args()
    cams, imus = collect_relevant_filters(args.src)
    for i, cam in enumerate(cams):
        process_camera_by_id(args.src, args.dst, cam, i)
    for i, imu in enumerate(imus):
        process_imu_by_id(args.src, args.dst, imu, i)


if __name__ == "__main__":
    main()
