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
import shutil
import cv2
import numpy as np
from pycocotools.coco import COCO
from data.CyC_db_interface import CyC_DataType, CyC_FilterType


def add_blockchain_descriptor(path2rovis, core_id, filter_id_images, filter_id_sem_seg, filter_id_objdet):
    if os.path.exists(os.path.join(path2rovis, "datablock_descriptor.csv")):
        print("Rovis blockchain descriptor file already exists")
        return
    with open(os.path.join(path2rovis, "datablock_descriptor.csv"), "w") as blockchain_desc:
        blockchain_desc.write("vision_core_id,filter_id,name,type,output_data_type,input_sources\n")
        if filter_id_images != -1:
            blockchain_desc.write("{},{},{},{},{},{}\n".format(
                core_id,
                filter_id_images,
                "Camera_{}".format(filter_id_images),
                CyC_FilterType.CyC_MONO_CAMERA_FILTER_TYPE,
                CyC_DataType.CyC_IMAGE,
                ""
            ))
        if filter_id_sem_seg != -1:
            blockchain_desc.write("{},{},{},{},{},{}\n".format(
                core_id,
                filter_id_sem_seg,
                "SemanticSegmentation_{}".format(filter_id_sem_seg),
                CyC_FilterType.CyC_SEMANTIC_SEGMENTATION_FILTER_TYPE,
                CyC_DataType.CyC_IMAGE,
                "{{{0}-{1}}}".format(core_id, filter_id_images)
            ))
        if filter_id_objdet != -1:
            blockchain_desc.write("{},{},{},{},{},{}\n".format(
                core_id,
                filter_id_objdet,
                "ObjectDetection_{}".format(filter_id_objdet),
                CyC_FilterType.CyC_OBJECT_DETECTOR_2D_FILTER_TYPE,
                CyC_DataType.CyC_2D_ROIS,
                "{{{0}-{1}}}".format(core_id, filter_id_images)
            ))


def add_timestamp_sync_file(path2rovis, filter_id_images, filter_id_semseg, filter_id_objdet):
    with open(os.path.join(path2rovis, "datastream_{}".format(filter_id_images), "data_descriptor.csv"), "r") as desc_img:
        lines_img = desc_img.readlines()

    with open(os.path.join(path2rovis, "sampling_timestamps_sync.csv"), "w") as ts_sync:
        ts_sync.write("timestamp_stop,datastream_{},datastream_{},datastream_{}\n".format(
            filter_id_images, filter_id_semseg, filter_id_objdet
        ))

        for line_img in lines_img:
            try:
                ts_stop_img = int(line_img.split(",")[0])
                ts_sync.write("{},{},{},{}\n".format(
                    ts_stop_img, ts_stop_img, ts_stop_img, ts_stop_img
                ))
            except ValueError:
                continue


def create_rovis_database(path2taco,
                          path2rovis,
                          filter_id_images=-1,
                          filter_id_sem_seg=-1,
                          filter_id_objdet=-1,
                          resize=(-1, -1)):


    sampling_time = 10
    start_timestamp = 1622720744706

    anns_file_path = path2taco + '/' + 'annotations.json'
    if not os.path.exists(anns_file_path):
        print("Taco dataset annotations.json not found at {}".format(path2taco))
        return False
    else:
        taco = COCO(anns_file_path)

    if not os.path.exists(path2taco):
        print("Mapillary dataset not found at {}".format(path2taco))
        return False

    if os.path.exists(path2rovis):
        print("Warning: Path {} already exists. Will not create folders...".format(path2rovis))
    else:
        os.mkdir(path2rovis)

    image_base_path = os.path.join(path2rovis, "datastream_{}".format(filter_id_images))
    sem_seg_base_path = os.path.join(path2rovis, "datastream_{}".format(filter_id_sem_seg))
    objdet_base_path = os.path.join(path2rovis, "datastream_{}".format(filter_id_objdet))

    if filter_id_images > 0:
        if os.path.exists(image_base_path):
            print("Warning: Path {} already exists. Will not create folders...".format(image_base_path))
        else:
            os.mkdir(image_base_path)
        if not os.path.exists(os.path.join(image_base_path, "samples")):
            os.mkdir(os.path.join(image_base_path, "samples"))
        if not os.path.exists(os.path.join(image_base_path, "samples", "0")):
            os.mkdir(os.path.join(image_base_path, "samples", "0"))
        if not os.path.exists(os.path.join(image_base_path, "samples", "0", "left")):
            os.mkdir(os.path.join(image_base_path, "samples", "0", "left"))

        images_to_write = taco.imgs
        num_imgs_to_write = len(images_to_write)

        with open(os.path.join(image_base_path, "data_descriptor.csv"), "w") as desc_f:
            desc_f.write("timestamp_start,timestamp_stop,sampling_time,left_file_path_0,right_file_path_0\n")
            for img_idx, img in enumerate(images_to_write.values()):
                img_path = os.path.join(path2taco, img['file_name'])
                print("Images: {}:{} Parsing frame: {}".format(
                    img_idx + 1,
                    num_imgs_to_write,
                    img['file_name']
                ))

                ts_stop = start_timestamp + img_idx * sampling_time
                ts_start = ts_stop - sampling_time
                dest_img_path = os.path.join(image_base_path, "samples", "0", "left", "{}.png".format(ts_stop))
                rel_img_path = "samples/0/left/{}.png".format(ts_stop)
                desc_f.write("{},{},{},{},\n".format(
                    ts_start, ts_stop, sampling_time, rel_img_path
                ))

                if resize[0] != -1 and resize[1] != -1:
                    img = cv2.imread(img_path)
                    h, w, c = img.shape
                    ratio_w = float(resize[0]) / w
                    ratio_h = float(resize[1]) / h
                    img_resize = (int(w * ratio_w),
                                  int(h * ratio_h))

                    img_resized = cv2.resize(img, img_resize)
                    cv2.imwrite(dest_img_path, img_resized)
                else:
                    shutil.copy(img_path, dest_img_path)

    if filter_id_sem_seg > 0:
        print("[##] Creating semantic segmentation datastream...")
        if os.path.exists(sem_seg_base_path):
            print("Warning: Path {} already exists. Will not create folders...".format(sem_seg_base_path))
        else:
            os.mkdir(sem_seg_base_path)
        if not os.path.exists(os.path.join(sem_seg_base_path, "samples")):
            os.mkdir(os.path.join(sem_seg_base_path, "samples"))
        if not os.path.exists(os.path.join(sem_seg_base_path, "samples", "0")):
            os.mkdir(os.path.join(sem_seg_base_path, "samples", "0"))
        if not os.path.exists(os.path.join(sem_seg_base_path, "samples", "0", "left")):
            os.mkdir(os.path.join(sem_seg_base_path, "samples", "0", "left"))
        if not os.path.exists(os.path.join(sem_seg_base_path, "samples", "1")):
            os.mkdir(os.path.join(sem_seg_base_path, "samples", "1"))
        if not os.path.exists(os.path.join(sem_seg_base_path, "samples", "1", "left")):
            os.mkdir(os.path.join(sem_seg_base_path, "samples", "1", "left"))

        images_to_write = taco.imgs
        num_imgs_sem = len(images_to_write)

        with open(os.path.join(sem_seg_base_path, "data_descriptor.csv"), "w") as desc_f:
            desc_f.write("timestamp_start,timestamp_stop,sampling_time,semantic,instances\n")
            with open(os.path.join(sem_seg_base_path, "framebased_data_descriptor.csv"), "w") as framebased:
                framebased.write("timestamp_stop,shape_id,cls,instance,points\n")
                for img in images_to_write.values():
                    crt_frame_id = img['id']
                    img_path = os.path.join(path2taco, img['file_name'])
                    print("SemSeg: {}:{} Parsing frame: {}".format(
                        crt_frame_id + 1, num_imgs_sem,
                        os.path.basename(img_path)
                    ))

                    ts_stop = start_timestamp + crt_frame_id * sampling_time
                    ts_start = ts_stop - 1

                    rel_img_path = "samples/0/left/{}.png".format(ts_stop)
                    rel_inst_path = "samples/1/left/{}.png".format(ts_stop)
                    desc_f.write("{},{},{},{},{}\n".format(
                        ts_start, ts_stop, sampling_time, rel_img_path, rel_inst_path
                    ))

                    dest_img_path = os.path.join(sem_seg_base_path, "samples", "0", "left", "{}.png".format(ts_stop))
                    dest_inst_img_path = os.path.join(sem_seg_base_path, "samples", "1", "left",
                                                      "{}.png".format(ts_stop))

                    annIds = taco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
                    anns_sel = taco.loadAnns(annIds)

                    w, h = img['width'], img['height']
                    ratio_w = float(resize[0]) / w
                    ratio_h = float(resize[1]) / h
                    img_shape = (int(h * ratio_h), int(w * ratio_w))

                    sem_seg_img = np.zeros(img_shape, dtype='uint8')
                    instance_img = np.zeros(img_shape, dtype='uint8')
                    instance_counter = []

                    for idx, annotation in enumerate(anns_sel):
                        cat_id = annotation['category_id'] + 1
                        inst_cont = instance_counter.count(cat_id) + 1
                        for seg in annotation['segmentation']:
                            coutours = np.array(seg)
                            coutours_x = coutours[0::2] * ratio_w
                            coutours_y = coutours[1::2] * ratio_h
                            coutours = [[x, y] for x, y in zip(coutours_x, coutours_y)]
                            coutours = np.array(coutours, dtype=np.int32)

                            seg_color = (int(cat_id), int(cat_id), int(cat_id))
                            cv2.fillPoly(sem_seg_img, pts=[coutours], color=seg_color)

                            instance_color = (int(inst_cont), int(inst_cont), int(inst_cont))
                            cv2.fillPoly(instance_img, pts=[coutours], color=instance_color)

                            points = '['
                            for c in coutours:
                                points += '[{}.0 {}.0]'.format(c[0], c[1])
                            points += ']'

                            framebased.write("{},{},{},{},{}\n".format(ts_stop,
                                                                idx + 1,
                                                                cat_id,
                                                                inst_cont,
                                                                points))
                        instance_counter.append(cat_id)
                    cv2.imwrite(dest_img_path, sem_seg_img)
                    cv2.imwrite(dest_inst_img_path, instance_img)

    if filter_id_objdet > 0:
        print("[##] Creating object detection datastream...")
        if os.path.exists(objdet_base_path):
            print("Warning: Path {} already exists. Will not create folders...".format(objdet_base_path))
        else:
            os.mkdir(objdet_base_path)

        images_to_write = taco.imgs
        num_imgs_to_write = len(images_to_write)

        with open(os.path.join(objdet_base_path, "data_descriptor.csv"), "w") as desc_f:
            with open(os.path.join(objdet_base_path, "framebased_data_descriptor.csv"), "w") as framebased_f:
                desc_f.write("timestamp_start,timestamp_stop,sampling_time,frame_id\n")
                framebased_f.write("frame_id,roi_id,cls,x,y,width,height\n")

                for img in images_to_write.values():
                    crt_frame_id = img['id']
                    annIds = taco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
                    anns_sel = taco.loadAnns(annIds)
                    print("ObjDet: {}:{} Parsing frame: {}".format(crt_frame_id + 1,
                                                                   num_imgs_to_write,
                                                                   img["id"]))

                    ts_stop = start_timestamp + sampling_time * crt_frame_id
                    ts_start = ts_stop - 1
                    desc_f.write("{},{},{},{}\n".format(
                        ts_start, ts_stop, sampling_time, crt_frame_id
                    ))

                    w, h = img['width'], img['height']
                    ratio_w = float(resize[0]) / w
                    ratio_h = float(resize[1]) / h

                    for idx, annotation in enumerate(anns_sel):
                        [bbox_x, bbox_y, bbox_w, bbox_h] = np.array(annotation['bbox']) * \
                                                           np.array([ratio_w, ratio_h, ratio_w, ratio_h])
                        bbox_id = idx
                        bbox_class = annotation['category_id'] + 1
                        framebased_f.write("{},{},{},{},{},{},{}\n".format(
                            crt_frame_id, bbox_id, bbox_class, bbox_x, bbox_y, bbox_w, bbox_h
                        ))


def generate_conf(path2taco, object_classes_folder_path):
    anns_file_path = path2taco + '/' + 'annotations.json'

    taco = COCO(anns_file_path)
    cat = taco.dataset['categories']
    new_dict = {}
    new_dict['background'] = {'id': -1, 'countable': 'True'}
    for item in cat:
        name = item.pop('name').lower().replace(" ", "_").replace("&", "and")
        item.pop('supercategory')
        item['countable'] = 'True'
        new_dict[name] = item

    file_path = os.path.join(object_classes_folder_path, 'object_classes_taco.conf')
    with open(file_path, 'w') as f:
        f.write('ObjectClasses:\n{\n')

        for key, val in new_dict.items():
            f.write(f'\t{key}:\n')
            f.write('\t{\n')
            id = val['id'] + 1
            cout = val['countable']
            f.write(f'\t\tID = {id}\n')
            f.write(f'\t\tCountable = {cout}\n')
            f.write('\t}\n')

        f.write('}')


if __name__ == '__main__':

    taco_data_path = r"D:\taco\data"
    rovis_path = r"D:\taco\rovis_taco_resize"
    resize = (640, 480)
    core_id = 1
    filter_id_images = 1
    filter_id_semseg = 2
    filter_id_objdet = 3
    object_classes_folder_path = r"C:\dev\RovisVision\etc\env"


    # generate_conf(path2taco=taco_data_path, object_classes_path=object_classes_path)


    create_rovis_database(path2taco=taco_data_path,
                          path2rovis=rovis_path,
                          filter_id_images=filter_id_images,
                          filter_id_sem_seg=filter_id_semseg,
                          filter_id_objdet=filter_id_objdet,
                          resize=resize)

    add_blockchain_descriptor(path2rovis=rovis_path,
                              core_id=core_id,
                              filter_id_images=filter_id_images,
                              filter_id_sem_seg=filter_id_semseg,
                              filter_id_objdet=filter_id_objdet)

    add_timestamp_sync_file(path2rovis=rovis_path,
                            filter_id_images=filter_id_images,
                            filter_id_semseg=filter_id_semseg,
                            filter_id_objdet=filter_id_objdet)