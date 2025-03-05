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
import os
import sys
import cv2
import torch
from torch.utils.data import DataLoader, Dataset

# dojo imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "dojo")))
from toolkit.vision.image_display_utils import drawSemanticSegmentation
from toolkit.vision.image_utils import imagebatch_2_tensor, decode_semseg
from toolkit.env.object_classes import ObjectClasses

class Dataset_Dummy(Dataset):
    def __init__(self, databases, hyperparams):
        self.databases = databases
        self.width = hyperparams['input_shape'][0][2]
        self.height = hyperparams['input_shape'][0][1]
        self.transforms = hyperparams['input_data_transforms']

    def __getitem__(self, item):
        # Dummy image and semantic segmentation
        img = np.ones((self.height, self.width, 3), dtype=np.float32)
        semantic = np.ones((self.height, self.width), dtype=np.uint8)

        # Convert to tensor and return
        return {
            "in_img": imagebatch_2_tensor([img], transforms=self.transforms)[0],
            "in_vector": torch.zeros((10)),
            "out_semantic": torch.from_numpy(semantic).long(),
            "out_classification_vector": torch.zeros((10)).float(),
            "out_regression_vector": torch.zeros((10)).float(),
        }

    def __len__(self):
        return 3  # Dummy length

    @staticmethod
    def collate_fn(batch):
        return torch.utils.data._utils.collate.default_collate(batch)


if __name__ == "__main__":
    object_classes = ObjectClasses("../../etc/env/object_classes_mapillary.conf")
    hyperparams = dict()
    hyperparams['input_shape'] = ((3, 320, 320),)
    hyperparams['input_data_transforms'] = []

    databases = []  # Dummy databases
    dataset = Dataset_Dummy(databases=databases, hyperparams=hyperparams)

    print('Dataset train has {} sets image-semseg.'.format(len(dataset)))

    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    max_imgs_to_disp = 5
    num_imgs_to_disp = min(max_imgs_to_disp, batch_size)

    for d in dataloader:
        imgs_disp = list()
        for idx in range(num_imgs_to_disp):
            img_orig = d["in_img"].cpu().detach().numpy()[idx]
            img_sem = d["out_semantic"].cpu().detach().numpy()[idx]
            img_orig = np.asarray(img_orig, dtype=np.float32) / 255.

            semseg_out = decode_semseg(img_sem, object_classes.colormap())
            final = drawSemanticSegmentation(img_orig, semseg_out)
            imgs_disp.append(final)

        final_image = cv2.hconcat(imgs_disp)
        cv2.imshow("Dataset_Segmentation2D", final_image)
        cv2.waitKey(0)
