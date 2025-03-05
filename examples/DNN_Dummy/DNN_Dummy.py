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
 * DNN_Dummy.py
 *
 *  Created on: 27.09.2021
 *      Author: Sorin Grigorescu
"""
from absl import app
import os
import logging
import io, libconf
import torch
from torch import nn
import onnx
import onnxruntime
from core.DNN_BaseClass import DNN_BaseClass
from toolkit.env.object_classes import ObjectClasses
from Dataset_Dummy import Dataset_Dummy

torch.manual_seed(1337)

# Initialize logger
logger = logging.getLogger(__name__)

class Model_Dummy(nn.Module):
    def __init__(self, input_shapes, output_shapes, device):
        super(Model_Dummy, self).__init__()
        self.conv = nn.Conv2d(input_shapes[0][0], output_shapes[0][0], 1).to(device)  # set up your layer here
        self.fc1 = nn.Linear(input_shapes[1][0], input_shapes[1][0]).to(device)  # set up first FC layer
        self.fc2 = nn.Linear(input_shapes[1][0], output_shapes[1][0]).to(device)
        self.fc3 = nn.Linear(input_shapes[1][0], output_shapes[2][0]).to(device)
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.device = device
        self.kernel_width = self.input_shapes[0][1] - self.output_shapes[0][1] + 1
        self.kernel_height = self.input_shapes[0][2] - self.output_shapes[0][2] + 1
        self.conv2 = nn.Conv2d(in_channels=self.output_shapes[0][0],
                               out_channels=self.output_shapes[0][0],
                               kernel_size=(self.kernel_width, self.kernel_height)).to(self.device)
        #
        # Input1 (image) ---> CNNLayer \                 / segmentation (img)
        #                               ---> concat --->
        # Input2 (vector) --> FCLayer  /                 \ classification (vector)
        #                                                 \ regression (vector)

    def forward(self, x1, x2):
        input1 = x1
        input2 = x2
        input1.to(self.device)
        input2.to(self.device)

        c1 = self.conv(input1)
        f1 = self.fc1(input2)

        # now we can reshape `c` and `f` to 2D and concat them
        combined = torch.cat((c1.view(c1.size(0), -1),
                              f1.view(f1.size(0), -1)), dim=1)

        # CRAPA
        interim = torch.split(tensor=combined, split_size_or_sections=self.output_shapes[0][0] * self.output_shapes[0][1] * self.output_shapes[0][2], dim=1)

        mid1 = combined[:, :-10]
        out1 = mid1.view(mid1.shape[0], self.output_shapes[0][0], self.input_shapes[0][1], self.input_shapes[0][2])
        out1 = self.conv2(out1)

        mid2 = combined[:, -10:]
        out2 = self.fc3(mid2)

        mid3 = combined[:, -10:]
        out3 = self.fc3(mid3)

        # out3 = torch.sigmoid(out3)
        out3 = out3.sigmoid()

        out = [out1, out2, out3]
        return out


class DNN_Dummy(DNN_BaseClass):
    def __init__(self, hyperparams_file):
        super(DNN_Dummy, self).__init__(hyperparams_file, Dataset_Dummy)
        self.Dummy_hyp = {}
        self.start_epoch = 0
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(self.hyperparams['device'])

        with io.open(self.hyperparams_file) as f:
            hyperparams_conf = libconf.load(f)
            self.Dummy_hyp = {
                'img_disp_size':            hyperparams_conf['Dummy_hyp']['img_disp_size'],
                'object_classes':           ObjectClasses(hyperparams_conf['Dummy_hyp']['object_classes']),
                'colormap':                 ObjectClasses(hyperparams_conf['Dummy_hyp']['object_classes']).colormap(),
                'predict_imgs_folder':      hyperparams_conf['Dummy_hyp']['predict_imgs_folder'],
            }

        # Construct DNN name by adding the number of classes
        self.dnn_name += '_classes-' + str(self.Dummy_hyp['object_classes'].num_classes)

        self.Model_End2End = Model_Dummy(output_shapes=self.hyperparams['output_shape'],
                                         input_shapes=self.hyperparams['input_shape'],
                                         device=self.hyperparams['device'])
        self.Model_End2End.to(self.hyperparams['device'])

        # Build the model
        self.build()

    def train_one_epoch(self, epoch):
        train_loss_on_epoch = 0
        self.Model_End2End.train()
        for batch_idx, batch_data in enumerate(self.dataloader_train):
            in_imgs = batch_data['in_img'].to(self.hyperparams['device'])
            in_vector = batch_data['in_vector'].to(self.hyperparams['device'])

            out_semantic_labels = batch_data['out_semantic'].to(self.hyperparams['device'])
            out_classification_vector_labels = batch_data['out_classification_vector'].to(self.hyperparams['device'])
            out_regression_vector_labels = batch_data['out_regression_vector'].to(self.hyperparams['device'])

            y = self.Model_End2End(in_imgs, in_vector)
            loss_1 = self.loss_fn(y[0], out_semantic_labels)
            loss_2 = self.loss_fn(y[1], out_classification_vector_labels)
            loss_3 = self.loss_fn(y[2], out_regression_vector_labels)

            train_loss_on_batch = loss_1 + loss_2 + loss_3
            train_loss_on_epoch += train_loss_on_batch.item()

        train_loss_on_epoch /= len(self.dataloader_train)
        print(f'Epoch {epoch}: train loss = {train_loss_on_epoch:.4f}')

    def predict(self, x=None):
        channels = 3
        in_tensor_width = self.hyperparams['input_shape'][0][2]
        in_tensor_height = self.hyperparams['input_shape'][0][1]
        input_img = torch.zeros((self.hyperparams['batch_size'], channels, in_tensor_height, in_tensor_width)).to(self.hyperparams['device'])
        input_vector = torch.zeros((self.hyperparams['batch_size'], 10)).to(self.hyperparams['device'])
        return self.Model_End2End(input_img, input_vector)

    def show_output(self, input_batch_tensor, output_tensor, wait_time=0):
        pass

    def predict_onnx(self):
        model_path = self.hyperparams['onnx_model_file']
        assert os.path.exists(model_path), "Onnx model path does not exist"

        logging.info("Predicting with ONNX model: {}".format(model_path))

        model = onnx.load(model_path)

        onnx.checker.check_model(model)
        onnx_inference_session = onnxruntime.InferenceSession(model_path)
        inputs = onnx_inference_session.get_inputs()

        input_img = torch.randn((self.hyperparams['batch_size'], *self.hyperparams['input_shape'][0]))
        input_vector = torch.ones((self.hyperparams['batch_size'], *self.hyperparams['input_shape'][1]))

        outputs = onnx_inference_session.run(None, {inputs[0].name: input_img.cpu().detach().numpy(),
                                                    inputs[1].name: input_vector.cpu().detach().numpy()})

        print("Something")
        print(outputs[0].shape)
        print(outputs[1])
        print(10*"-")
        print(outputs[2])

        return outputs


def tu_DNN_Dummy(_argv):
    dnn = DNN_Dummy("DNN_Dummy.conf")

    y = dnn.predict()
    y_onnx = dnn.predict_onnx()
    print(20 * "=")


if __name__ == '__main__':
    try:
        app.run(tu_DNN_Dummy)
    except SystemExit:
        pass