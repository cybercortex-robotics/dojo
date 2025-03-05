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
 * DNN_BaseClass.cpp
 *
 *  Created on: 23.07.2021
 *      Author: Sorin Grigorescu
 *
 * Naming conventions:
 *  DNN_...:    class derived from DNN_BaseClass
 *  Module_...: class derived from torch.nn.Module
"""

from absl import app, flags
import numpy as np
import random
import logging
from abc import ABC, abstractmethod
import os
import io, libconf
import torch
from torch.utils.tensorboard import SummaryWriter
from toolkit.dl.lr_scheduler import PolyLR
from toolkit.dl.checkpoint_utils import load_weights, load_last_checkpoint

# Initialize logger
logger = logging.getLogger(__name__)

if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

class DNN_BaseClass(ABC):
    dnn_name: str
    Datasets_Train: None
    Datasets_Validation: None
    Datasets_Test: None
    hyperparams_file: str
    is_build: bool = False
    tensorboard_writer: None

    def __init__(self, hyperparams_file, dataset_type):
        # The DNN's name is the name of the hyperparameters file without the extension
        self.dnn_name = os.path.splitext(os.path.basename(hyperparams_file))[0]
        self.hyperparams_file = hyperparams_file
        self.Model_End2End = None
        self.is_build = False
        self.Dataset_Type = dataset_type
        self.start_epoch = 0
        self.optimizer = None
        self.lr_scheduler = None

        # Parse hyperparameters file
        assert os.path.isfile(self.hyperparams_file), f'{self.hyperparams_file} does not exist'
        with io.open(self.hyperparams_file) as f:
            hyperparams_conf = libconf.load(f)

            # Parse common hyperparameters
            self.hyperparams = {
                'device':                   hyperparams_conf['Common']['device'],
                'is_training':              hyperparams_conf['Common']['is_training'],
                'epochs':                   hyperparams_conf['Common']['epochs'],
                'learning_rate':            hyperparams_conf['Common']['learning_rate'],
                'momentum':                 hyperparams_conf['Common']['momentum'],
                'batch_size':               hyperparams_conf['Common']['batch_size'],
                'input_data':               hyperparams_conf['Common']['input_data'],
                'input_shape':              hyperparams_conf['Common']['input_shape'],
                'output_data':              hyperparams_conf['Common']['output_data'],
                'output_shape':             hyperparams_conf['Common']['output_shape'],
                'shuffle':                  hyperparams_conf['Common']['shuffle'],
                'num_workers':              hyperparams_conf['Common']['num_workers'],
                'train_split':              hyperparams_conf['Common']['train_split'],
                'eval_freq':                hyperparams_conf['Common']['eval_freq'],
                'optimizer':                hyperparams_conf['Common']['optimizer'],
                'random_seed':              hyperparams_conf['Common']['random_seed'],
                'plot_architecture':        hyperparams_conf['Common']['plot_architecture'],
                'view_predictions':         hyperparams_conf['Common']['view_predictions'],
                'ckpts_dir':                hyperparams_conf['Common']['ckpts_dir'],
                'ckpt_freq':                hyperparams_conf['Common']['ckpt_freq'],
                'load_last_ckpt':           hyperparams_conf['Common']['load_last_ckpt'],
                'onnx_export':              hyperparams_conf['Common']['onnx_export'],
                'onnx_opset_version':       hyperparams_conf['Common']['onnx_opset_version'],
                'onnx_model_file':          hyperparams_conf['Common']['onnx_model_file'],
                'load_pretrained_weights':  hyperparams_conf['Common']['load_pretrained_weights'],
                'pretrained_weights':       hyperparams_conf['Common']['pretrained_weights'],
                'input_data_transforms':    hyperparams_conf['Common']['input_data_transforms'],
            }

            # Seed for reproducibility in random number generation
            if self.hyperparams['random_seed'] > -1:
                self.set_random_seed()

            # Check common hyperparameters
            if self.hyperparams['load_last_ckpt']:
                assert os.path.isdir(self.hyperparams['ckpts_dir'])

            assert 0 < self.hyperparams['train_split'] <= 1
            assert len(self.hyperparams['input_data']) == len(self.hyperparams['input_data_transforms'])

            # Parse the train, validation and test datasets
            self.Datasets_Train = self.parse_datasets(hyperparams_conf, 'Datasets_Train')
            self.Datasets_Validation = self.parse_datasets(hyperparams_conf, 'Datasets_Validation')
            self.Datasets_Test = self.parse_datasets(hyperparams_conf, 'Datasets_Test')

            # Initialize Tensorboard
            self.tensorboard_writer = SummaryWriter()

    def build(self):
        # Init data parallelism
        # self.Model_End2End = torch.nn.DataParallel(self.Model_End2End)

        if self.hyperparams['load_pretrained_weights']:
            path = os.path.abspath(self.hyperparams['pretrained_weights'])
            load_weights(self.Model_End2End, path)

        if self.hyperparams['load_last_ckpt']:
            ckpts_path = os.path.abspath(os.path.join(os.getcwd(), self.hyperparams['ckpts_dir']))
            self.start_epoch = load_last_checkpoint(ckpts_folder_path=ckpts_path,
                                                    model=self.Model_End2End)

        # save_submodel(self.Model_End2End, 0, 42, "EncDec_Backbone.pth")
        # save_submodel(self.Model_End2End, 43, 43, "EncDec_Head_classes_150.pth")
        # save_submodel(self.Model_End2End, "model_head.model", "EncDec_Head_classes_11.pth")

        self.is_build = True

        # Plot DNN architecture
        if self.hyperparams['plot_architecture']:
            self.plot_arch()

        # Load datasets
        self.load_datasets()

        # Configure network
        if self.hyperparams['is_training']:
            # Configure optimizer
            if self.hyperparams['optimizer'] == 'sgd':
                self.optimizer = torch.optim.SGD(self.Model_End2End.parameters(),
                                                 lr=self.hyperparams['learning_rate'],
                                                 momentum=self.hyperparams['momentum'])
            elif self.hyperparams['optimizer'] == 'adamw':
                self.optimizer = torch.optim.AdamW(self.Model_End2End.parameters(),
                                                   lr=self.hyperparams['learning_rate'])
            else:
                logger.error('Optimizer {} not supported. Exiting.'.format(self.hyperparams['optimizer']))

            self.lr_scheduler = PolyLR(optimizer=self.optimizer, pow=0.9, max_iter=20000)

            self.train()
        else:
            self.Model_End2End.eval()

            # Export to ONNX
            if self.hyperparams['onnx_export']:
                self.export_onnx(self.hyperparams['onnx_model_file'])

    # @abstractmethod
    def load_datasets(self):
        if self.hyperparams['is_training']:
            # Train dataset -------------------------------------------------------------------------------------------
            dataset_train = self.Dataset_Type(databases=self.Datasets_Train, hyperparams=self.hyperparams)

            if len(dataset_train) == 0:
                logger.error("Empty train dataset!")
                exit(0)

            # Validation dataset --------------------------------------------------------------------------------------
            if self.Datasets_Validation is not None:
                dataset_validation = self.Dataset_Type(databases=self.Datasets_Validation, hyperparams=self.hyperparams)
            else:
                if self.hyperparams['train_split'] < 1:
                    num_train_samples = int(len(dataset_train) * self.hyperparams['train_split'])
                    num_validation_samples = len(dataset_train) - num_train_samples
                    dataset_train, dataset_validation = \
                        torch.utils.data.random_split(dataset_train, [num_train_samples, num_validation_samples])
                else:
                    dataset_validation = None

            # Batching with batch_size and using our own collate_fn
            rank = -1
            sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) if rank != -1 else None
            self.dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                                batch_size=self.hyperparams['batch_size'],
                                                                shuffle=self.hyperparams['shuffle'],
                                                                num_workers=self.hyperparams['num_workers'],
                                                                sampler=sampler_train,
                                                                pin_memory=False,
                                                                collate_fn=self.Dataset_Type.collate_fn)
            logger.info("Train dataset: {} samples {} batches".format(len(dataset_train), len(self.dataloader_train)))

            if dataset_validation:
                rank = -1
                sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val) if rank != -1 else None
                self.dataloader_val = torch.utils.data.DataLoader(dataset_validation,
                                                                  batch_size=self.hyperparams['batch_size'],
                                                                  shuffle=self.hyperparams['shuffle'],
                                                                  num_workers=self.hyperparams['num_workers'],
                                                                  sampler=sampler_val,
                                                                  pin_memory=False,
                                                                  collate_fn=self.Dataset_Type.collate_fn)

                logger.info(
                    "Validation dataset: {} samples {} batches".format(len(dataset_validation), len(self.dataloader_val)))
        else:
            # Test dataset -------------------------------------------------------------------------------------------
            if self.Datasets_Test is not None:
                dataset_test = self.Dataset_Type(databases=self.Datasets_Test, hyperparams=self.hyperparams)
                rank = -1
                sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test) if rank != -1 else None
                self.dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                                   batch_size=self.hyperparams['batch_size'],
                                                                   shuffle=self.hyperparams['shuffle'],
                                                                   num_workers=self.hyperparams['num_workers'],
                                                                   sampler=sampler_test,
                                                                   pin_memory=False,
                                                                   collate_fn=self.Dataset_Type.collate_fn)

    def train(self):
        for epoch in range(self.start_epoch, self.hyperparams["epochs"]):
            self.train_one_epoch(epoch=epoch)

    @abstractmethod
    def train_one_epoch(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, input_batch_tensor):
        pass

    @abstractmethod
    def show_output(self, input_batch_tensor, output_tensor):
        pass

    @abstractmethod
    def predict_onnx(self, image_folder):
        pass

    def parse_datasets(self, hyperparams_conf, datasets_type):
        if len(hyperparams_conf[datasets_type]) > 0:
            datasets = list()
            for set in hyperparams_conf[datasets_type]:
                dataset = {
                    'path': hyperparams_conf[datasets_type][set]['path'],
                    'keys_samples': hyperparams_conf[datasets_type][set]['keys_samples'],
                    'keys_labels': hyperparams_conf[datasets_type][set]['keys_labels']
                }

                if not os.path.exists(dataset['path']):
                    logger.error("Dataset path '{}' does not exist. Exiting ...".format(dataset['path']))
                    exit()

                datasets.append(dataset)
            return datasets
        else:
            logger.warning("No {} found in '{}'".format(datasets_type, self.hyperparams_file))
            return None

    def plot_arch(self):
        from torchview import draw_graph
        # Random test sample and forward propagation
        x = torch.rand((self.hyperparams['batch_size'], *self.hyperparams['input_shape'][0]),
                       dtype=torch.float32).to(self.hyperparams['device'])
        draw_graph(model=self.Model_End2End,
                   input_data=x,
                   graph_name=self.dnn_name,
                   directory='.',
                   save_graph=True,
                   hide_inner_tensors=True,
                   hide_module_functions=True,
                   expand_nested=True)

    # Export the model to ONNX format
    def export_onnx(self, file_path_onnx):
        assert self.is_build

        x = list()
        for i in range(len(self.hyperparams['input_shape'])):
            shape = self.hyperparams['input_shape'][i]
            dtype = torch.float32
            if self.hyperparams['input_data'][i] == 'CyC_INT':
                dtype = torch.int64
            x.append(torch.rand((self.hyperparams['batch_size'], *shape), dtype=dtype).to(self.hyperparams['device']))
        x = tuple(x)

        logger.info("Exporting ONNX model to '{}'".format(file_path_onnx))
        torch.onnx.export(self.Model_End2End,  # model to export
                          x,  # input tensor
                          file_path_onnx,  # output file
                          opset_version=self.hyperparams['onnx_opset_version'],  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to optimize the model by folding constants
                          input_names=["input"],  # the name of the input tensor
                          output_names=["output"],  # the name of the output tensor
                          dynamic_axes={"input": {0: "batch_size"},  # variable-length axes
                                        "output": {0: "batch_size"}})
        # torch.onnx.export(self.Model_End2End,  # model to export
        #                   x,  # input tensor
        #                   file_path_onnx,  # output file
        #                   opset_version=self.hyperparams['onnx_opset_version'],
        #                   do_constant_folding=True,  # whether to optimize the model by folding constants
        #                   input_names=["kpts0", "kpts1"],  # the name of the input tensor
        #                   output_names=["matches0", "mscores0"],  # the name of the output tensor
        #                   dynamic_axes={
        #                       "kpts0": {1: "num_keypoints0"},
        #                       "kpts1": {1: "num_keypoints1"},
        #                       "matches0": {0: "num_matches0"},
        #                       "mscores0": {0: "num_matches0"}, })
        logger.info("ONNX model '{}' exported successfully".format(file_path_onnx))

    def set_random_seed(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def tu_DNN_BaseClass(_argv):
    class DNN_DerivedClass(DNN_BaseClass):
        def __init__(self, hyperparams_file):
            super(DNN_DerivedClass, self).__init__(hyperparams_file)

        def build(self):
            pass

        def get_configuration(self):
            pass

        def plot_arch(self):
            pass

        def train(self):
            pass

        def train_one_epoch(self, *args, **kwargs):
            pass

        def predict(self, input_batch_tensor):
            pass

        def draw_output(self, input_batch_tensor, output_tensor):
            pass

        def show_output(self, input_batch_tensor, output_tensor):
            pass

        def export_onnx(self, file_path_onnx):
            pass

        def predict_onnx(self, image_folder):
            pass

    DNN_DerivedClass("DNN_Dummy/DNN_Dummy.conf")


if __name__ == '__main__':
    try:
        app.run(tu_DNN_BaseClass)
    except SystemExit:
        pass
