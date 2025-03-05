"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import sys
import os
import tensorflow as tf
from tensorflow import saved_model
from tensorflow.core.framework import types_pb2
from tensorflow.python.tools import saved_model_utils
import tensorflow.keras as keras
#import global_config
#CFG = global_config.cfg


def _get_inputs_tensor_info_from_meta_graph_def(meta_graph_def,
                                                signature_def_key):
    """Gets TensorInfo for all inputs of the SignatureDef.
    Returns a dictionary that maps each input key to its TensorInfo for the given
    signature_def_key in the meta_graph_def
    Args:
      meta_graph_def: MetaGraphDef protocol buffer with the SignatureDef map to
          look up SignatureDef key.
      signature_def_key: A SignatureDef key string.
    Returns:
      A dictionary that maps input tensor keys to TensorInfos.
    """
    return meta_graph_def.signature_def[signature_def_key].inputs


def _get_outputs_tensor_info_from_meta_graph_def(meta_graph_def,
                                                 signature_def_key):
    """Gets TensorInfos for all outputs of the SignatureDef.
    Returns a dictionary that maps each output key to its TensorInfo for the given
    signature_def_key in the meta_graph_def.
    Args:
      meta_graph_def: MetaGraphDef protocol buffer with the SignatureDefmap to
      look up signature_def_key.
      signature_def_key: A SignatureDef key string.
    Returns:
      A dictionary that maps output tensor keys to TensorInfos.
    """
    return meta_graph_def.signature_def[signature_def_key].outputs


def _print_tensor_info(tensor_info, indent=0):
    """Prints details of the given tensor_info.
    Args:
      tensor_info: TensorInfo object to be printed.
      indent: How far (in increments of 2 spaces) to indent each line output
    """
    indent_str = '  ' * indent

    def in_print(s):
        print(indent_str + s)

    in_print('    dtype: ' +
             {value: key
              for (key, value) in types_pb2.DataType.items()}[tensor_info.dtype])
    # Display shape as tuple.
    if tensor_info.tensor_shape.unknown_rank:
        shape = 'unknown_rank'
    else:
        dims = [str(dim.size) for dim in tensor_info.tensor_shape.dim]
        shape = ', '.join(dims)
        shape = '(' + shape + ')'
    in_print('    shape: ' + shape)
    in_print('    name: ' + tensor_info.name)


def _show_inputs_outputs(saved_model_dir, tag_set, signature_def_key, indent=0):
    """Prints input and output TensorInfos.
    Prints the details of input and output TensorInfos for the SignatureDef mapped
    by the given signature_def_key.
    Args:
      saved_model_dir: Directory containing the SavedModel to inspect.
      tag_set: Group of tag(s) of the MetaGraphDef, in string format, separated by
          ','. For tag-set contains multiple tags, all tags must be passed in.
      signature_def_key: A SignatureDef key string.
      indent: How far (in increments of 2 spaces) to indent each line of output.
    """
    meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir,
                                                          tag_set)
    inputs_tensor_info = _get_inputs_tensor_info_from_meta_graph_def(
        meta_graph_def, signature_def_key)
    outputs_tensor_info = _get_outputs_tensor_info_from_meta_graph_def(
        meta_graph_def, signature_def_key)

    indent_str = '  ' * indent

    def in_print(s):
        print(indent_str + s)

    in_print('The given SavedModel SignatureDef contains the following input(s):')
    for input_key, input_tensor in sorted(inputs_tensor_info.items()):
        in_print('  inputs[\'%s\'] tensor_info:' % input_key)
        _print_tensor_info(input_tensor, indent+1)

    in_print('The given SavedModel SignatureDef contains the following '
             'output(s):')
    for output_key, output_tensor in sorted(outputs_tensor_info.items()):
        in_print('  outputs[\'%s\'] tensor_info:' % output_key)
        _print_tensor_info(output_tensor, indent+1)

    in_print('Method name is: %s' %
             meta_graph_def.signature_def[signature_def_key].method_name)


def group_model_names(output_tensor_names):
    groups = dict()
    groups["yolov3_tiny"] = list()
    groups["yolov3"] = list()
    groups["lanenet"] = list()
    groups["unet"] = list()
    groups["box_estimation"] = list()

    for o in output_tensor_names:
        if "yolov3_tiny" in o:
            groups["yolov3_tiny"].append((o, output_tensor_names[o]))
        elif "yolov3" in o:
            groups["yolov3"].append((o, output_tensor_names[o]))
        elif "lanenet" in o:
            groups["lanenet"].append((o, output_tensor_names[o]))
        elif "unet" in o:
            groups["unet"].append((o, output_tensor_names[o]))
        elif "model_1" in o:
            groups["box_estimation"].append((o, output_tensor_names[o]))
        elif "box_estimation" in o:
            groups["box_estimation"].append((o, output_tensor_names[o]))
        else:
            raise NotImplementedError("Unknown Head")
    return groups


def get_output_index(output_name, output_tensor_info):
    idx = 0
    ret_idx = -1
    for o in output_tensor_info:
        if output_name == o:
            ret_idx = idx
        idx += 1
    if ret_idx == -1:
        raise Exception("Could not write HeadsOutputIDs")
    return ret_idx


def create_config(model_path, output_config_path):
    tag_sets = saved_model_utils.get_saved_model_tag_sets(model_path)
    tag_set = ""
    for tag in tag_sets:
        if "serve" in tag:
            tag_set = tag
            print("Found tag set \"serve\"")

    if tag_set == "":
        print("Could not find \"serve\" tag set in model. Exiting...")
        sys.exit(-1)

    if isinstance(tag_set, list):
        tag_set = tag_set[0]

    meta_graph_def = saved_model_utils.get_meta_graph_def(model_path, tag_set)
    signature_key_def = ""
    for k in dict(meta_graph_def.signature_def).keys():
        if "serving" in k:
            signature_key_def = k
    if signature_key_def == "":
        print("Could not find serving_default signature key def. Exiting...")
        sys.exit(-1)

    inputs_tensor_info = _get_inputs_tensor_info_from_meta_graph_def(
        meta_graph_def, signature_key_def)
    outputs_tensor_info = _get_outputs_tensor_info_from_meta_graph_def(
        meta_graph_def, signature_key_def)

    print("Inputs:")
    for k in inputs_tensor_info:
        print("layer_name = {0}".format(k))
        print("v = {0}".format(inputs_tensor_info[k].name))
    print("="*30)
    print("Outputs:")
    for k in outputs_tensor_info:
        print("layer_name = {0}".format(k))
        print("v = {0}".format(outputs_tensor_info[k].name))

    try:
        model = keras.models.load_model(model_path)
        print("Model inputs:")
        input_names_ordered = list()
        output_names_ordered = list()
        for i in model.inputs:
            input_names_ordered.append(i.name.split(":")[0])
            print(i.name.split(":")[0])
        print("Model outputs:")
        for o in model.outputs:
            output_names_ordered.append(o.name.split("/")[0])
            print(o.name.split("/")[0])

        print("=" * 30)
        for o in output_names_ordered:
            print(o + "=>" + outputs_tensor_info[o].name)

        for i in input_names_ordered:
            print(i + "=>" + inputs_tensor_info[i].name)
    except Exception as e:
        print("Could not load model. Network metadata will be written in the read order")
        input_names_ordered = inputs_tensor_info
        output_names_ordered = outputs_tensor_info

    with open(output_config_path, "w") as f:
        f.write("ModelParameters:\n")
        f.write("{\n")

        f.write("    ModelPath = \"{0}\"\n".format(os.path.abspath(model_path)))

        #f.write("    ConfigProto = \"{0}\"\n".format(os.path.abspath(os.path.join(CFG.BASE.PATH,
        #                                                                          "core",
        #                                                                          "etc",
        #                                                                          "dnn",
        #                                                                          "config.pb"))))
        f.write("    InputShape: (\n")
        # print(input_names_ordered["input_image_shape"].tensor_shape)
        for i in input_names_ordered:
            if len(input_names_ordered) > 1:
                f.write("    {\n")
            if len(inputs_tensor_info[i].tensor_shape.dim) == 4:  # Image
                f.write("        {0}, /*{1}*/\n".format(inputs_tensor_info[i].tensor_shape.dim[1].size, "width"))
                f.write("        {0}, /*{1}*/\n".format(inputs_tensor_info[i].tensor_shape.dim[2].size, "height"))
                f.write("        {0}, /*{1}*/\n".format(inputs_tensor_info[i].tensor_shape.dim[3].size, "channels"))
            else:
                raise NotImplementedError("Unknown input type and / or shape")
            if len(input_names_ordered) > 1:
                f.write("    }\n")
        f.write("    )\n")

        f.write("    Heads: (\n")
        heads = group_model_names(outputs_tensor_info)
        for head in heads:
            if len(heads[head]):
                f.write("        \"{}\",\n".format(head))
        f.write("    )\n")

        f.write("    HeadsOutputNumber: {\n")
        for head in heads:
            if len(heads[head]):
                f.write("        {} = {},\n".format(head, len(heads[head])))
        f.write("    }\n")

        f.write("    HeadsOutputIDs: {\n")
        for head in heads:
            for o, output_name in heads[head]:
                o_name = o.strip()
                if "model_1" in o_name:
                    o_name += "_0"
                elif "box_estimation" in o_name:
                    o_name += "_0"
                elif "yolov3_tiny" in o_name:
                    if "_" not in o_name:
                        o_name += "_0"
                elif "yolov3" in o_name:
                    if "_" not in o_name:
                        o_name += "_0"
                elif "unet" in o_name:
                    if "_" not in o_name:
                        o_name += "_0"
                elif "lanenet" in o_name:
                    if "_" not in o_name:
                        o_name += "_0"
                else:
                    raise Exception("Head name not recognised")
                f.write("        {} = {},\n".format(o_name, get_output_index(o, outputs_tensor_info)))
        f.write("    }\n")

        # Write inputs
        f.write("    InputLayerNames: (\n")
        for i in input_names_ordered:
            f.write("        \"{0}\",\n".format(inputs_tensor_info[i].name.split(":")[0]))
        f.write("    )\n")
        f.write("    InputLayerIndex: (\n")
        for i in input_names_ordered:
            f.write("        {0}, /*{1}*/\n".format(inputs_tensor_info[i].name.split(":")[1], i))
        f.write("    )\n")

        # Write outputs
        f.write("    OutputLayerNames: (\n")
        for o in output_names_ordered:
            f.write("        \"{0}\",\n".format(outputs_tensor_info[o].name.split(":")[0]))
        f.write("    )\n")
        f.write("    OutputLayerIndex: (\n")
        for o in output_names_ordered:
            f.write("        {0}, /*{1}*/\n".format(outputs_tensor_info[o].name.split(":")[1], o))
        f.write("    )\n")
        f.write("}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_configuration.py /path/2/network /path/2/output/conf/file")
        exit(1)
    create_config(sys.argv[1], sys.argv[2])
