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
"""
from absl import app
import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


class Model_Backbone_EfficientNet(nn.Module):
    r"""
    EfficientNet backbone.
    The forward function returns a dictionary of cached features for a list of requested layers,
    or the output of the last layer, if no requested layers are specifically returned.
    """
    def __init__(self,
                 backbone_type: str = "efficientnet-b0",
                 cached_layers: list = None,
                 device: str = "cpu"):
        super(Model_Backbone_EfficientNet, self).__init__()
        self.backbone_type = backbone_type
        self.cached_layers = cached_layers
        self.device = device
        self.model = None

        if backbone_type == "efficientnet-b0":
            self.model = EfficientNet.from_pretrained('efficientnet-b0').to(self.device)
        else:
            raise ValueError("Backbone type '{}' is not supported.".format(backbone_type))

        # Freeze the layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Count the number of layers in the model
        num_layers = 0
        for child in self.model.children():
            if isinstance(child, nn.Sequential):
                num_layers += len(child)
            else:
                num_layers += 1

        # Check if the requested layers are available
        if self.cached_layers is not None:
            assert max(self.cached_layers) < num_layers, \
                "Requested layer is not available in the '{}' backbone. " \
                "Requested layer: {}, Maximum layers: {}".format(self.__class__.__name__, max(self.cached_layers), len(self.model))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, dict):
        for k, child in self.model.named_children():
            print(k)
        # if self.cached_layers is not None:
        #     cached_outputs = {None: None} if None in self.cached_layers else dict()
        #     layer_id: int = 0
        #     for _, child in self.model.named_children():
        #         x = child(x)
        #         if layer_id in self.cached_layers:
        #             cached_outputs[layer_id] = x
        #         layer_id += 1
        #     return cached_outputs
        # else:
        #     return self.model(x)


def tu_Model_Backbone_EfficientNet(_argv):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_shape = (1, 3, 320, 320)
    cached_layers = [4, 6]
    model = Model_Backbone_EfficientNet(backbone_type="efficientnet-b0",
                                        cached_layers=cached_layers,
                                        device=device)

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

    from torchview import draw_graph
    draw_graph(model=model,
               input_data=x,
               graph_name="efficientnet",
               directory='.',
               save_graph=True,
               hide_inner_tensors=True,
               hide_module_functions=True,
               expand_nested=True)

    # print("Num cached features:", len(y))


if __name__ == '__main__':
    try:
        app.run(tu_Model_Backbone_EfficientNet)
    except SystemExit:
        pass
