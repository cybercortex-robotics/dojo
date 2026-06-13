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
 * data_transformations.py
 *
 *  Created on: 06.10.2021
 *      Author: Sorin Grigorescu
"""

import numpy as np
import cv2
import logging
import torchvision

# Initialize logger
logger = logging.getLogger(__name__)

def transform_imagesbatch(batch, transforms):
    """
    Applies transformations directly on image data (mainly using opencv operations)
    :param batch: input batch
    :param transforms: vector of transformations
    :return: transformed batch
    """
    if transforms is None:
        return batch

    for i in range(len(transforms)):
        entry = transforms[i]
        if hasattr(entry, 'keys'):
            # Hydra/OmegaConf-style entry, e.g. {'rgb2bgr': {...}}
            for type_name in entry:
                if type_name.lower() == "rgb2bgr":
                    for idx2 in range(len(batch)):
                        batch[idx2] = cv2.cvtColor(batch[idx2], cv2.COLOR_RGB2BGR)
        else:
            for transform in entry:
                if transform[0] == "RGB2BGR":
                    for idx2 in range(len(batch)):
                        batch[idx2] = cv2.cvtColor(batch[idx2], cv2.COLOR_RGB2BGR)
    return batch

def untransform_imagesbatch(batch, transforms):
    """
    Applies inverse transformations directly on image data (mainly using opencv operations)
    :param batch: input batch
    :param transforms: vector of transformations
    :return: transformed batch
    """
    for i in range(len(transforms)):
        entry = transforms[i]
        if hasattr(entry, 'keys'):
            # Hydra/OmegaConf-style entry, e.g. {'rgb2bgr': {...}}
            for type_name in entry:
                if type_name.lower() == "rgb2bgr":
                    for idx2 in range(len(batch)):
                        batch[idx2] = cv2.cvtColor(batch[idx2], cv2.COLOR_BGR2RGB)
        else:
            for transform in entry:
                if transform[0] == "RGB2BGR":
                    for idx2 in range(len(batch)):
                        batch[idx2] = cv2.cvtColor(batch[idx2], cv2.COLOR_BGR2RGB)
    return batch

def transform_tensorbatch(batch, transforms):
    """
    Applies torch transformations on the input batch tensor
    :param batch: input batch
    :param transforms: vector of transformations
    :return: transformed batch
    """
    if transforms is None:
        return batch
    
    for i in range(len(transforms)):
        entry = transforms[i]
        if hasattr(entry, 'keys'):
            # Hydra/OmegaConf-style entry, e.g. {'normalize': {'mean': [...], 'std': [...]}}
            for type_name, params in entry.items():
                if type_name.lower() == "normalize":
                    mean = params['mean']
                    std = params['std']
                    batch = torchvision.transforms.Normalize(mean=mean, std=std)(batch)
        else:
            for transform in entry:
                if transform[0] == "Normalize":
                    mean = transform[1]
                    std = transform[2]
                    batch = torchvision.transforms.Normalize(mean=mean, std=std)(batch)
    return batch

def untransform_tensorbatch(batch, transforms):
    """
    Applies inverse torch transformations on the input batch tensor
    :param batch: input batch
    :param transforms: vector of transformations
    :return: transformed batch
    """
    for i in range(len(transforms)):
        entry = transforms[i]
        if hasattr(entry, 'keys'):
            # Hydra/OmegaConf-style entry, e.g. {'normalize': {'mean': [...], 'std': [...]}}
            for type_name, params in entry.items():
                if type_name.lower() == "normalize":
                    mean = np.asarray(params['mean'])
                    std = np.asarray(params['std'])
                    invTrans = torchvision.transforms.Compose([
                        torchvision.transforms.Normalize(mean=np.full(len(mean), 0.), std=1./std),
                        torchvision.transforms.Normalize(mean=-mean, std=np.full(len(std), 1.)),
                    ])
                    batch = invTrans(batch)
        else:
            for transform in entry:
                if transform[0] == "Normalize":
                    mean = np.asarray(transform[1])
                    std = np.asarray(transform[2])
                    invTrans = torchvision.transforms.Compose([
                        torchvision.transforms.Normalize(mean=np.full(len(mean), 0.), std=1./std),
                        torchvision.transforms.Normalize(mean=-mean, std=np.full(len(std), 1.)),
                    ])
                    batch = invTrans(batch)
    return batch
