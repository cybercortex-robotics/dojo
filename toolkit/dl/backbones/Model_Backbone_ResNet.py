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
 *  Created on: 17.02.2022
 *      Author: Sorin Grigorescu
"""
from absl import app
import torch
import torch.nn as nn
from torchvision import models


class Model_Backbone_ResNet(nn.Module):
    r"""
    ResNet backbone.
    The forward function returns a dictionary of cached features for a list of requested layers,
    or the output of the last layer, if no requested layers are specifically returned.
    """
    def __init__(self,
                 backbone_type: str = "resnet18",
                 cached_layers: list = None,
                 pretrained: bool = True):
        super(Model_Backbone_ResNet, self).__init__()
        self.backbone_type = backbone_type
        self.cached_layers = cached_layers

        if self.backbone_type == "resnet18":
            if pretrained:
                self.model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
            else:
                self.model = models.resnet18(weights=None)
        elif self.backbone_type == "resnet50":
            self.model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        else:
            raise ValueError("Backbone type '{}' is not supported.".format(backbone_type))

        # Remove last non-cached layers
        layers = list(self.model.children())
        if cached_layers is not None:
            layers = layers[:max(self.cached_layers)-(len(layers)-1)]   # [:-2]
        else:
            layers = layers[:-2]
        self.model = nn.Sequential(*layers)

        if pretrained is True:
            # Freeze the layers
            for param in self.model.parameters():
                param.requires_grad = False

        # Check if the requested layers are available
        if self.cached_layers is not None:
            assert max(self.cached_layers) < len(self.model), \
                "Requested layer is not available in the '{}' backbone. " \
                "Requested layer: {}, Maximum layers: {}".format(self.__class__.__name__, max(self.cached_layers), len(self.model))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, dict):
        if self.cached_layers is not None:
            cached_outputs = {None: None} if None in self.cached_layers else dict()
            layer_id: int = 0
            for _, child in self.model.named_children():
                x = child(x)
                # print(layer_id, x.shape)
                if layer_id in self.cached_layers:
                    cached_outputs[layer_id] = x
                layer_id += 1
            return cached_outputs
        else:
            return self.model(x)


def tu_Model_Backbone_ResNet(_argv):
    device = "cuda"
    input_shape = (1, 3, 320, 320)
    cached_layers = [5, 6, 7]
    backbone_type = "resnet18"
    model = Model_Backbone_ResNet(backbone_type=backbone_type,
                                  cached_layers=cached_layers).to(device)

    x = torch.zeros(input_shape, dtype=torch.float32).to(device)
    y = model(x)

    # Parse the model state dict
    # state_dict_model = model.state_dict()
    # id = 0
    # for k in state_dict_model:
    #     print(id, ":", k, "\t shape:", state_dict_model[k].shape)
    #     id += 1
    from torchsummary import summary
    summary(model, input_size=input_shape[1:])

    from torchview import draw_graph
    draw_graph(model=model,
               input_data=x,
               graph_name=model.__class__.__name__ + "_" + backbone_type,
               directory='.',
               save_graph=True,
               hide_inner_tensors=True,
               hide_module_functions=True,
               expand_nested=True)

    print("Num cached features:", len(y))


if __name__ == '__main__':
    try:
        app.run(tu_Model_Backbone_ResNet)
    except SystemExit:
        pass
