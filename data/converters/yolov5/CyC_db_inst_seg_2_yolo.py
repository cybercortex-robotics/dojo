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
import shutil
from data.CyC_db_interface import *
from global_config import cfg as CFG
from toolkit.object_classes import ObjectClasses


DEBUG = True

object_classes = ObjectClasses(CFG.CyC_INFERENCE.OBJECT_CLASSES_PATH)


def update_timestamp_sync(rovis_db_path, yolo_filter_id, sample_filter_id):
    if not os.path.exists(os.path.join(rovis_db_path, "sampling_timestamps_sync.csv")):
        print("[###] ERROR: Rovis sampling timestamps sync file does not exist")
        exit(1)

    with open(os.path.join(rovis_db_path, "sampling_timestamps_sync.csv"), "r") as ts_sync_file:
        lines = ts_sync_file.readlines()

    if "datastream_{}".format(yolo_filter_id) in lines[0]:
        print("[###] WARNING: datastream_{} already present in timestamp sync file. File won't be modified".format(yolo_filter_id))
        return

    idx_samples = 0
    try:
        idx_samples = lines[0].strip().split(",").index("datastream_{}".format(sample_filter_id))
    except ValueError:
        print("[###] ERROR: Sample datastream_{} not found in timestamp_sync file")
        exit(1)

    lines[0] = lines[0].strip() + ",datastream_{}\n".format(yolo_filter_id)
    for idx in range(1, len(lines)):
        line = lines[idx].strip().split(",")
        ts_stop = int(line[idx_samples])
        lines[idx] = lines[idx].strip() + ",{}".format(ts_stop)

    with open(os.path.join(rovis_db_path, "sampling_timestamps_sync.csv"), "w") as ts_sync_file:
        for line in lines:
            ts_sync_file.write(line.strip() + "\n")


def update_blockchain_descriptor(rovis_db_path, yolo_filter_id, sample_filter_id):
    if not os.path.exists(rovis_db_path):
        print("[###] ERROR: Rovis Database path {} invalid".format(rovis_db_path))
        exit(1)

    if not os.path.exists(os.path.join(rovis_db_path, "datablock_descriptor.csv")):
        print("[###] ERROR: datablock_descriptor.csv file not found at path {}".format(
            rovis_db_path
        ))
        exit(1)

    with open(os.path.join(rovis_db_path, "datablock_descriptor.csv")) as blockchain_desc:
        lines = blockchain_desc.readlines()

    sample_core_id = 1
    for line in lines:
        if "filter_id" in line:
            continue
        filter_id = int(line.strip().split(",")[1])
        if filter_id == yolo_filter_id:
            sample_core_id = int(line.strip().split(",")[0])
            lines.remove(line)
            break
        if filter_id == sample_filter_id:
            sample_core_id = int(line.strip().split(",")[0])

    lines.append("{},{},ObjectDetection2D,{},{},{{{}-{}}}\n".format(
        sample_core_id, yolo_filter_id,
        CyC_FilterType.CyC_OBJECT_DETECTOR_2D_FILTER_TYPE,
        CyC_DataType.CyC_2D_ROIS,
        sample_core_id,
        sample_filter_id
    ))

    with open(os.path.join(rovis_db_path, "datablock_descriptor.csv"), "w") as blockchain_desc:
        for line in lines:
            blockchain_desc.write(line.strip() + "\n")


def prepare_yolo_files(rovis_db_path, yolo_filter_id, sample_filter_id):
    if not os.path.exists(rovis_db_path):
        print("[###] ERROR: Rovis Database path {} invalid".format(rovis_db_path))
        exit(1)

    if not os.path.exists(os.path.join(rovis_db_path, "datastream_{}".format(sample_filter_id))):
        print("[###] ERROR: Samples filter path {} does not exist".format(
            os.path.join(rovis_db_path, "datastream_{}".format(sample_filter_id))
        ))
        exit(1)

    if not os.path.exists(os.path.join(rovis_db_path, "datastream_{}".format(sample_filter_id), "data_descriptor.csv")):
        print("[###] ERROR: Descriptor csv from {} does not exist".format(
            os.path.join(rovis_db_path, "datastream_{}".format(sample_filter_id))
        ))
        exit(1)

    if os.path.exists(os.path.join(rovis_db_path, "datastream_{}".format(yolo_filter_id))):
        print("[###] Warning: Folder {} already exists. Deleting...".format(
            os.path.join(rovis_db_path, "datastream_{}".format(yolo_filter_id))
        ))
        shutil.rmtree(os.path.join(rovis_db_path, "datastream_{}".format(yolo_filter_id)), ignore_errors=True)

    os.mkdir(os.path.join(rovis_db_path, "datastream_{}".format(yolo_filter_id)))

    with open(os.path.join(rovis_db_path, "datastream_{}".format(sample_filter_id), "data_descriptor.csv"), "r") as sample_f:
        lines_samples = sample_f.readlines()

    frame_id = 0
    with open(os.path.join(rovis_db_path, "datastream_{}".format(yolo_filter_id), "data_descriptor.csv"), "w") as desc_f:
        desc_f.write("timestamp_start,timestamp_stop,sampling_time,frame_id\n")
        for line in lines_samples:
            if "timestamp_start" in line:
                continue
            ts_start = int(line.strip().split(",")[0])
            ts_stop = int(line.strip().split(",")[1])
            sampling_time = int(line.strip().split(",")[2])
            desc_f.write("{},{},{},{}\n".format(
                ts_start, ts_stop, sampling_time, frame_id
            ))
            frame_id += 1


def generate(rovis_db, yolo_filter_id, object_classes_objects, object_classes_regions, map_objects_fcn=None):
    """
    Parse the Rovis database and generate bounding boxes for yolo detector
    """
    samples_files = list()
    samples_metadata = list()
    labels_files = list()
    labels_semseg_files = list()

    if len(rovis_db['keys_samples']) != len(rovis_db['keys_labels']):
        print("[##] ERROR: keys_samples should be same length as keys_labels")
        exit(1)

    if not os.path.exists(object_classes_objects):
        print("[##] ERROR: Please provide valid object_classes for objects")
        exit(1)

    if not os.path.exists(object_classes_regions):
        print("[##] ERROR: Please provide valid object_classes for regions")
        exit(1)

    if len(rovis_db['keys_samples']) != 1:
        print("[##] ERROR: generate function supports just one sample + label at once")
        exit(1)

    if object_classes_regions != object_classes_objects:
        if map_objects_fcn is None:
            print("[##] Object classes for regions differs from object classes for objects. "
                  "Please provide map function between classes")
            exit(1)

    obj_cls_objects = ObjectClasses(object_classes_objects)

    max_idx_objects = obj_cls_objects.get_class_max_index()

    dataset = rovis_db
    print("[#] Reading dataset located at path {}".format(dataset['path']))
    if not os.path.exists(dataset['path']):
        print("[##] ERROR: Database folder {} does not exist. Skipping")
        exit(1)

    frame_id = 0
    for s, l in zip(dataset['keys_samples'], dataset['keys_labels']):
        desc_s = os.path.join(dataset['path'], "datastream_{}".format(s), "data_descriptor.csv")
        desc_l = os.path.join(dataset['path'], "datastream_{}".format(l), "data_descriptor.csv")
        with open(desc_s, "r") as f_s:
            with open(desc_l, "r") as f_l:
                lines_samples = f_s.readlines()
                lines_labels = f_l.readlines()

        # Iterate over each semseg / inst seg sample
        for idx in range(1, len(lines_samples)):
            try:
                ts_start = int(lines_samples[idx].split(",")[0])
                ts_stop = int(lines_samples[idx].split(",")[1])
                sampling_time = int(lines_samples[idx].split(",")[2])
                img_relative_path = lines_samples[idx].split(",")[3]

            except IndexError:
                # frame_id should increment at every frame
                frame_id += 1
                continue
            img_path = os.path.join(dataset['path'], "datastream_{}".format(s), img_relative_path)
            if not os.path.exists(img_path) or os.path.isdir(img_path):
                # frame_id should increment at every frame
                frame_id += 1
                continue
            try:
                label_relative_path = lines_labels[idx].strip().split(",")[4]  # instance segmentation
            except IndexError:
                # frame_id should increment at every frame
                frame_id += 1
                continue
            label_img_path = os.path.join(dataset['path'], "datastream_{}".format(l), label_relative_path)
            if not os.path.exists(label_img_path) or os.path.isdir(label_img_path):
                # frame_id should increment at every frame
                frame_id += 1
                continue
            try:
                label_semseg_relative_path = lines_labels[idx].strip().split(",")[3]  # semantic segmentation
            except IndexError:
                # frame_id should increment at every frame
                frame_id += 1
                continue
            label_semseg_img_path = os.path.join(dataset['path'], "datastream_{}".format(l), label_semseg_relative_path)
            if not os.path.exists(label_semseg_img_path) or os.path.isdir(label_semseg_img_path):
                # frame_id should increment at every frame
                frame_id += 1
                continue
            samples_metadata.append({'ts_start': ts_start, 'ts_stop': ts_stop, 'sampling_time': sampling_time,
                                     'frame_id': frame_id, 'key_sample': s, 'key_label': l})
            samples_files.append(img_path)
            labels_files.append(label_img_path)
            labels_semseg_files.append(label_semseg_img_path)
            frame_id += 1

    print("[##] Preparing yolo directory structure and descriptor file")
    prepare_yolo_files(rovis_db_path=dataset['path'],
                       yolo_filter_id=yolo_filter_id,
                       sample_filter_id=dataset['keys_samples'][0])

    print("[##] Updating blockchain descriptor file")
    update_blockchain_descriptor(rovis_db_path=dataset['path'],
                                 yolo_filter_id=yolo_filter_id,
                                 sample_filter_id=dataset['keys_samples'][0])

    print("[##] Updating timestamp sync file")
    update_timestamp_sync(rovis_db_path=dataset['path'],
                          yolo_filter_id=yolo_filter_id,
                          sample_filter_id=dataset['keys_samples'][0])

    print("[##] Parsing instance segmentation images")
    idx = 0
    with open(os.path.join(dataset['path'], "datastream_{}".format(yolo_filter_id), "framebased_data_descriptor.csv"), "w") as frame_based:

        frame_based.write("frame_id,roi_id,cls,x,y,width,height\n")
        for img, l_file, semseg_file, meta in zip(samples_files, labels_files, labels_semseg_files, samples_metadata):
            # if DEBUG:
            print("[##] Image number {}:{}".format(idx + 1, len(labels_files)))
            img_l = cv2.imread(l_file)
            img_s = cv2.imread(semseg_file)
            img_o = cv2.imread(img)

            all_rgb_codes = img_s.reshape(-1, img_s.shape[-1])
            unique_rgbs = np.unique(all_rgb_codes, axis=0)

            all_instance_codes = img_l.reshape(-1, img_l.shape[-1])
            unique_instances = np.unique(all_instance_codes, axis=0)

            unique_semseg_classes = np.unique(unique_rgbs)
            unique_instances_clases = np.unique(unique_instances)
            for color in unique_semseg_classes:
                if color == object_classes.background_class:
                    continue
                sem_seg_img = np.zeros((img_o.shape[0], img_o.shape[1], 3))
                if DEBUG:
                    print("[###] color={}, code={}, countable={}".format(color,
                                                                         object_classes.get_name_by_index(color),
                                                                         object_classes.is_countable(color)))
                sem_seg_img[np.where((img_s == color).all(axis=2))] = [True, True, True]

                roi_id = 0
                for instance in unique_instances_clases:
                    if instance == 0:
                        continue
                    # inst_seg_img = np.zeros((img_o.shape[0], img_o.shape[1], 3))
                    inst_seg_img = np.logical_and(sem_seg_img, img_l == instance)
                    inst_seg_img = 255 * np.asarray(inst_seg_img, dtype=np.uint8)
                    all_zero = np.all((inst_seg_img == 0))
                    if all_zero:
                        continue
                    if DEBUG:
                        print("[####] Instance {} of object class {}, frame_id {}".format(instance,
                                                                                          object_classes.get_name_by_index(color),
                                                                                          meta['frame_id']))

                    img_cpy = inst_seg_img.copy()
                    img_cpy = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2GRAY)
                    x, y, w, h = cv2.boundingRect(img_cpy)
                    if map_objects_fcn is not None:
                        color = map_objects_fcn(color, object_classes_regions, object_classes_objects)
                    if color > max_idx_objects or color < 0:
                        if DEBUG:
                            print("[###] Skipping bbox {},{},{},{},{},{},{} due to invalid class".format(
                                meta["frame_id"],
                                roi_id,
                                color,
                                x, y, w, h
                            ))
                        continue
                    if DEBUG:
                        print("[###] Writing bbox {},{},{},{},{},{},{}".format(
                            meta["frame_id"],
                            roi_id,
                            color,
                            x, y, w, h
                        ))
                    frame_based.write("{},{},{},{},{},{},{}\n".format(
                        meta["frame_id"],
                        roi_id,
                        color,
                        x, y, w, h
                    ))

                    roi_id += 1

                    if DEBUG:
                        pass
                        # cv2.imshow("Instance", inst_seg_img)
                        # img_cpy = cv2.cvtColor(img_cpy, cv2.COLOR_GRAY2BGR)
                        # cv2.rectangle(img_cpy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # cv2.imshow("Colored bbox", img_cpy)
                        # cv2.waitKey(20)
            idx += 1

    # Copy object classes file
    print("Copying object classes file {} to {}".format(object_classes_objects,
                                                        os.path.join(dataset['path'],
                                                                     "datastream_{}".format(yolo_filter_id),
                                                                     "object_classes.conf")
                                                        )
          )
    shutil.copy(object_classes_objects,
                os.path.join(dataset['path'], "datastream_{}".format(yolo_filter_id), "object_classes.conf"))


def nuscenes_region_to_object(cls, region_object_classes_file, objects_object_classes_file):
    """
    Mapper function.
    It should map classes from NuScenes_regions to NuScenes_objects
    """
    obj_cls_r = ObjectClasses(region_object_classes_file)
    obj_cls_o = ObjectClasses(objects_object_classes_file)

    if cls > obj_cls_o.get_class_max_index():
        print("[###] Warning: Class {} does not have correspondent in object classes file. Skipping...".format(cls))

    return cls


if __name__ == "__main__":
    rovis_database = {
        'path': r"C:\data\RovisDatabases\Driving\dataset_driving",
        'keys_samples': [1],
        'keys_labels': [10]
    }

    object_classes_reg = "C:/dev/src/RovisLab/RovisVision/etc/env/classes_NuScenes_regions.conf"
    object_classes_objects = "C:/dev/src/RovisLab/RovisVision/etc/env/classes_NuScenes_objects.conf"

    generate(rovis_db=rovis_database,
             yolo_filter_id=11,
             object_classes_objects=object_classes_objects,
             object_classes_regions=object_classes_reg,
             map_objects_fcn=nuscenes_region_to_object)

