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
 * Model_Backbone_Generator.py
 *
 *  Created on: 26.02.2022
 *      Author: Sorin Grigorescu

 https://github.com/jeremyfix/pytorch_feature_extraction/blob/master/dltools.py
 https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/05/27/extracting-features.html
"""
from absl import app
import torch
import torch.nn as nn
from torchvision import models


class Model_Backbone_MobileNet(nn.Module):
    r"""
    MobileNet backbone.
    The forward function returns a dictionary of cached features for a list of requested layers,
    or the output of the last layer, if no requested layers are specifically returned.
    """
    def __init__(self,
                 backbone_type: str = "mobilenet_v2",
                 cached_layers: list = None):
        super(Model_Backbone_MobileNet, self).__init__()
        self.backbone_type = backbone_type
        self.cached_layers = cached_layers

        if self.backbone_type == "mobilenet_v2":
            self.model = models.mobilenet_v2(weights="MobileNet_V2_Weights.IMAGENET1K_V1").features
        else:
            raise ValueError("Backbone type '{}' is not supported.".format(backbone_type))

        # Remove the last FC layer
        # self.model = nn.Sequential(*list(self.model.children())[:-1])

        # Freeze the layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Check if the requested layers are available
        if self.cached_layers is not None:
            assert max(self.cached_layers) < len(self.model), \
                "Requested layer is not available in the '{}' backbone. " \
                "Requested layer: {}, Maximum layers: {}".format(self.__class__.__name__, max(self.cached_layers), len(self.model))

    def forward(self, x: torch.Tensor):
        if self.cached_layers is not None:
            cached_outputs = {None: None} if None in self.cached_layers else dict()
            for layer_id, child in enumerate(self.model):
                x = child(x)
                # print(layer_id, x.shape)
                if layer_id in self.cached_layers:
                    cached_outputs[layer_id] = x
            return cached_outputs
        else:
            return self.model(x)


def tu_Model_Backbone_MobileNet(_argv):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_shape = (1, 3, 320, 320)
    cached_layers = [4, 6]
    model = Model_Backbone_MobileNet(backbone_type="mobilenet_v2",
                                     cached_layers=cached_layers).to(device)

    x = torch.zeros(input_shape, dtype=torch.float32).to(device)
    y = model(x)

    # Parse the model state dict
    # state_dict_model = model.state_dict()
    # id = 0
    # for k in state_dict_model:
    #     print(id, ":", k, "\t shape:", state_dict_model[k].shape)
    #     id += 1
    # from torchsummary import summary
    # summary(model, input_size=input_shape[1:])

    # from torchview import draw_graph
    # draw_graph(model=model,
    #            input_data=x,
    #            graph_name="Model_Backbone_MobileNet",
    #            directory='.',
    #            save_graph=True,
    #            hide_inner_tensors=True,
    #            hide_module_functions=True,
    #            expand_nested=True)

    # print("Num cached features:", len(y))

    # torch.onnx.export(model,  # model to export
    #                   x,  # input tensor
    #                   "mobilenetV2.onnx",  # output file
    #                   opset_version=12,  # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to optimize the model by folding constants
    #                   input_names=["input"],  # the name of the input tensor
    #                   output_names=["output"],  # the name of the output tensor
    #                   dynamic_axes={"input": {0: "batch_size"},  # variable-length axes
    #                                 "output": {0: "batch_size"}})


if __name__ == '__main__':
    try:
        app.run(tu_Model_Backbone_MobileNet)
    except SystemExit:
        pass
