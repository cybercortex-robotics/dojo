"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import os
import argparse
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np


def set_memory_growth(allow_growth):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, allow_growth)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print("Failed to set memory growth:")
            print(e)
    else:
        print("WARNING: No gpus available.")


def set_memory_limit(limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)
    else:
        print("WARNING: No gpus available.")


def create_config_proto(args):
    if args.enable_growth is not None:
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=args.enable_growth)
    elif args.memory_limit is not None:
        gpu_options = tf.compat.v1.GPUOptions(memory_limit_mb=args.memory_limit)
    else:
        gpu_options = tf.compat.v1.GPUOptions()

    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    serialized = config.SerializeToString()
    return serialized


def main():
    parser = argparse.ArgumentParser(description="Optimization using TensorRT")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="SavedModel location")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Location where the optimized SavedModel should be saved")
    parser.add_argument("-i", "--input_file", type=str, required=False, help="Input for neural network as numpy format")
    parser.add_argument("--wsz", "--workspace_size", type=int, default=3150000000, help="Maximum workspace size in bytes for TensorRT")
    cgrp = parser.add_mutually_exclusive_group(required=False)
    cgrp.add_argument("-g", "--enable_growth", action="store_true", help="Allow growing memory for TF")
    cgrp.add_argument("-l", "--memory_limit", type=int, help="Set max limit for TF memory usage in MB")
    qgrp = parser.add_mutually_exclusive_group(required=False)
    qgrp.add_argument("--fp32", default=False, action="store_true")
    qgrp.add_argument("--fp16", default=False, action="store_true")
    qgrp.add_argument("--int8", default=False, action="store_true")

    args = parser.parse_args()

    if args.enable_growth is not None:
        set_memory_growth(args.enable_growth)
    if args.memory_limit is not None:
        set_memory_limit(args.memory_limit)

    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)

    if args.int8:
        params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32)
    elif args.fp16:
        params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16)
    else:
        params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32)

    params = params._replace(maximum_cached_engines=16)
    params = params._replace(max_workspace_size_bytes=args.wsz)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=args.model_path, conversion_params=params)
    converter.convert()

    if args.input_file:
        input_data = np.load(args.input_file)

        def data_gen_func():
            def _gen_func():
                yield (input_data,)

            return _gen_func

        converter.build(input_fn=data_gen_func())

    converter.save(args.output_path)

    config_proto = create_config_proto(args)
    with open(os.path.join(args.output_path, "config.pb"), "w+") as f:
        f.write(config_proto)
        print("Configuration for TF was written into", os.path.join(args.output_path, "config.pb"))


if __name__ == '__main__':
    print("Tensorflow Version:", tf.__version__)
    main()
