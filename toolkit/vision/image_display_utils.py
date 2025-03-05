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
 * image_display_utils.py
 *
 *  Created on: 27.08.2021
 *      Author: Sorin Grigorescu
"""

import numpy as np
import cv2
import torch
from ..vision.image_utils import nchw_2_nhwc
from ..dl.data_transformations import untransform_tensorbatch, untransform_imagesbatch
from ..vision.yolo_utils import xywh2xyxy

def drawObjectsOnBatches(input_images_batch, output_targets, object_classes, conf_threshold, input_data_transforms=None):
    assert len(input_images_batch) != 0

    # Apply inverse data transforms to input tensor
    if input_data_transforms is not None:
        input_images_batch = untransform_tensorbatch(input_images_batch, input_data_transforms)

    # Convert input tensor to numpy data
    if isinstance(input_images_batch, torch.Tensor):
        input_images_batch = input_images_batch.cpu().float().numpy()
    if isinstance(output_targets, torch.Tensor):
        output_targets = output_targets.cpu().numpy()
    if isinstance(input_images_batch, list):
        input_images_batch = np.asarray(input_images_batch).astype(np.float)
    if isinstance(output_targets, list):
        output_targets = np.asarray(output_targets)

    # Apply inverse data transforms to input numpy data
    if input_data_transforms is not None:
        input_images_batch = untransform_imagesbatch(input_images_batch, input_data_transforms)

    # Scaling
    w = input_images_batch.shape[2]
    h = input_images_batch.shape[1]
    scale = 1
    if (w < 640 or h < 480) and len(output_targets) > 0:
        scale *= 3

    tl = 3
    tf = max(tl - 1, 1)  # font thickness
    colormap = object_classes.colormap()
    batch_plotted = list()
    for i in range(len(input_images_batch)):
        img = input_images_batch[i]
        img = cv2.resize(img, (w * scale, h * scale))

        if len(output_targets) > 0:
            image_targets = output_targets[output_targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6])
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 6  # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)
            tracking_ids = -1 * torch.ones(classes.shape)  # image_targets[:, 7]

            for j in range(len(boxes)):
                if tracking_ids[j] >= 0:
                    label = '%s %d' % (object_classes.get_name_by_index(int(classes[j])), tracking_ids[j])
                else:
                    label = '%s' % (object_classes.get_name_by_index(int(classes[j])))
                info_conf = '%.2f' % (conf[j]) if conf is not None else ''

                box = boxes[j]
                color = (float(colormap[classes[j]][0]), float(colormap[classes[j]][1]), float(colormap[classes[j]][2]))
                c1, c2 = (int(box[0] * scale), int(box[1] * scale)), (int(box[2] * scale), int(box[3] * scale))
                cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                sub_img = np.asarray(img[c1[1]:c2[1], c1[0]:c2[0]]).astype(float)
                colored_rect = np.full(sub_img.shape, color, dtype=float)
                res = cv2.addWeighted(sub_img, 0.8, colored_rect, 0.5, 0.2)
                img[c1[1]:c2[1], c1[0]:c2[0]] = res  # Putting the image back to its position

                cv2.rectangle(img, (c1[0] - tl + 1, c1[1] - tf * 18), (c2[0] + tl - 1, c1[1]), color, thickness=-1, lineType=cv2.LINE_AA)
                cv2.putText(img, label, (c1[0], c1[1] - 2), 1, tl * 0.8, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
                cv2.putText(img, info_conf, (c1[0], c2[1] - 2), 1, tl * 0.8, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

            img = cv2.resize(img, (w, h))

        batch_plotted.append(img)
    return np.asarray(batch_plotted)


def drawSemanticSegmentation(img, segmentation):
    """ Draw semantic segmentation image """
    # Transpose channels
    if (img.shape[0] <= 3 or img.shape[1] <= 3) and img.shape[2] > 3:
        img = img.transpose(1, 2, 0)
    if (segmentation.shape[0] <= 3 or segmentation.shape[1] <= 3) and segmentation.shape[2] > 3:
        segmentation = segmentation.transpose(1, 2, 0)
    # segmentation = segmentation * 255

    # img = np.array(img, dtype=np.float32)
    if np.max(img) > 2:
        img /= 255.

    if np.max(segmentation) > 2:
        segmentation /= 255.

    if img.shape != segmentation.shape:
        img = cv2.resize(img, (segmentation.shape[1], segmentation.shape[0]), interpolation=cv2.INTER_NEAREST)

    return cv2.addWeighted(np.asarray(img).astype(np.float32), 0.9, np.asarray(segmentation).astype(np.float32), 0.4, 0.0) * 255.


def drawSemanticSegmentationOnBatches(input_images_batch, output_tensor):
    input_images_batch = nchw_2_nhwc(input_images_batch)

    batch_viz = list()
    for k in range(len(output_tensor)):
        img = drawSemanticSegmentation(input_images_batch[k], output_tensor[k])
        batch_viz.append(img)

    # Convert list to numpy array
    batch_viz = np.asarray(batch_viz)
    return batch_viz

def plot_colormap(object_classes, colormap):
    final_img = np.ones((1, 400, 3), np.uint8) * 255
    for idx in range(object_classes.num_classes):
        cls_name = object_classes.get_name_by_index(idx)
        cls_color = colormap[idx]
        aux_img = np.ones((35, 400, 3), np.uint8) * 255
        aux_img[:, 0:30] = cls_color
        aux_img = cv2.putText(aux_img, cls_name, (35, 27), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        final_img = cv2.vconcat([final_img, aux_img])
    return final_img

def drawKeypointsOnBatches(input_images_batch, score, uv, input_data_transforms=None):
    pass


def drawVoxels():
    """ Draws 3D voxels """
    pass

def drawCorrespondencesOnBatches(prev_images, curr_images, prev_pts, curr_pts):
    assert prev_images.shape[0] == curr_images.shape[0], "Previous and current batch size must be the same"
    assert prev_images.shape[1] == curr_images.shape[1], "Previous and current channels number must be the same"
    assert prev_images.shape[2] == curr_images.shape[2], "Previous and current height must be the same"
    assert prev_images.shape[3] == curr_images.shape[3], "Previous and current width must be the same"
    assert prev_pts.shape[0] == curr_pts.shape[0], "Previous and current points number must be the same"

    B, _, H, W = prev_images.shape

    if isinstance(prev_images, torch.Tensor):
        prev_images = prev_images.cpu().float().numpy()
    if isinstance(curr_images, torch.Tensor):
        curr_images = curr_images.cpu().numpy()
    if isinstance(prev_pts, torch.Tensor):
        prev_pts = prev_pts.cpu().float().numpy()
    if isinstance(curr_pts, torch.Tensor):
        curr_pts = curr_pts.cpu().numpy()

    # un-normalise
    if np.max(prev_images[0]) <= 1:
        prev_images *= 255
    if np.max(curr_images[0]) <= 1:
        curr_images *= 255

    batch_corresp = list()
    for k in range(B):
        prev_img = np.transpose(prev_images[k], (1, 2, 0)).astype(np.uint8)
        curr_img = np.transpose(curr_images[k], (1, 2, 0)).astype(np.uint8)
        disp = cv2.hconcat([prev_img, curr_img])

        for i in range(0, prev_pts.shape[0]):
            pt_source_image = (int(prev_pts[i, 0]), int(prev_pts[i, 1]))
            pt_target_image = (int(curr_pts[i, 0] + W), int(curr_pts[i, 1]))
            cv2.line(disp, pt_source_image, pt_target_image, (0, 255, 0), 1)

            # debug = disp.copy()
            # cv2.line(debug, pt_source_image, pt_target_image, (0, 255, 0), 1)
            # cv2.imshow('corresp', debug)
            # cv2.waitKey(0)

        batch_corresp.append(disp)
    return cv2.vconcat(batch_corresp)
