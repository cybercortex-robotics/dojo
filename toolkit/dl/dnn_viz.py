"""
Copyright (c) 2026 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing,
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

"""
 * dnn_viz.py
 *
 *  Created on: 18.06.2026
 *      Author: Sorin Grigorescu
"""

import torch
import logging

logger = logging.getLogger(__name__)


def plot_arch(model,
              input_shapes,
              graph_name="dnn_arch",
              device="cpu",
              output_dir=".",
              tb_writer=None):
    """Render the model architecture as a Graphviz PDF via torchview.

    Args:
        model:        torch.nn.Module to visualise.
        input_shapes: List of shape tuples, one per model input head,
                      e.g. [(3, 224, 224)] or [(3, 224, 224), (1, 64, 64)].
                      Batch dimension is added automatically.
        graph_name:   Base filename for the saved graph (no extension).
        device:       Device string or torch.device for the dummy tensors.
        output_dir:   Directory in which to save the rendered graph file.
        tb_writer:    Optional torch.utils.tensorboard.SummaryWriter.
                      When provided, the graph is also logged to TensorBoard.
    """
    try:
        from torchview import draw_graph
    except ImportError:
        logger.error("torchview is not installed. Run: pip install torchview")
        return

    dummy_inputs = [
        torch.rand(1, *shape, dtype=torch.float32).to(device)
        for shape in input_shapes
    ]
    input_data = dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs

    draw_graph(
        model=model,
        input_data=input_data,
        graph_name=graph_name,
        directory=output_dir,
        save_graph=True,
        hide_inner_tensors=True,
        hide_module_functions=True,
        expand_nested=True,
    )
    logger.info("Architecture graph saved to '%s/%s'", output_dir, graph_name)

    if tb_writer is not None:
        try:
            tb_writer.add_graph(model, tuple(dummy_inputs))
            logger.info("Architecture graph logged to TensorBoard")
        except Exception as exc:
            logger.warning("TensorBoard add_graph failed: %s", exc)
