# How to use the optimizer
The optimizer is provided with name `optimize_model.py`. It is a basic Python script that takes advantage of the `trt_compiler` inside `tensorflow` to optimize nodes inside neural networks combining them into `TensorRT` nodes.

To take advantage of the optimizer you need a NVIDIA graphics card. The optimizer will not run on CPU.

## Usage
```
usage: optimize_model.py [-h] -m MODEL_PATH -o OUTPUT_PATH [-i INPUT_FILE]
                         [--wsz WSZ] [-g | -l MEMORY_LIMIT]
                         [--fp32 | --fp16 | --int8]
```
* `-h` prints helpul information about arguments
* `-m`, `--model_path` is the path to the SavedModel file
* `-o`, `--output_path` is the output directory where the optimized model will be saved
* `-i`, `--input_file` is the input file used for inference on the neural network
* `--wsz` specifies the workspace memory limit for the TensorRT optimizer
* `-g` sets to whether allow or not TF to use incremental memory usage (recommended)
* `-l`, `--memory_limit` sets the maximum amount of memory TF should use in MB. This is mutually exclusive with `-g`
* `--fp32`, `--fp16`, `--int8` optimizes the neural network and sets the data type to `float32` (default), `float16`, or `int8`. `float16` and `int8` are not available on any graphics card. Options are mutually exclusive.


## How-to
Example usage:
```
python3 optimize_model.py -m normal_model_path -o optimized_model_path -i test_input.npy -g
```

The argument `-i test_input.npy` is optional, but it is recommended. The content of `test_input.npy` is a batch of inputs for the neural network. Only 1 input is enough. The format of the file should be `npy`. For example, for `YOLOv3`, this file can be generated as follows:
```
image = np.random.random((1, 416, 416, 3)).astype(np.float32)
np.save("test_input.npy", image)
```

Options `--fp32`, `--fp16` and `--int8` specifies the data type of the resulting neural network model. `--fp32` is usually the default and does not change anything. The other two might not be available on any graphics card. `--int8` in particular needs calibration. Without calibration, the output of the neural network might degrade.

Since TF allocates 95% of VRAM, it is recommended to specify either `-g` or `-l <value>` to limit its usage. TensorRT requires quite a lot of memory, depending on the complexity of the neural network.

Errors regarding memory are solved by the above indications, sometimes in combination with `--wsz <value>` option. This controls the maximum amount of memory available to TensorRT. By default (in this script), the value is `3150000000`, which is over around 3GB. The value was set to be used for `YOLOv3`. For simpler networks, the amount can be lowered, and for more complex neural network the amount should be increased. Keep in mind that these values are dependant on the VRAM of the computer.

Example of memory errors:
```
    Cuda error in file src/implicit_gemm.cu at line 585: out of memory

or

    cuBLAS Error in initializeCommonContext: 3 (Could not initialize cublas, please check cuda installation.)

or

    Cudnn Error in initializeCommonContext: 4 (Could not initialize cudnn, please check cudnn installation.)

and even

    Executor failed to create kernel. Not found: No registered 'CombinedNonMaxSuppression' OpKernel for 'GPU' devices
```

These errors are solved by specifying `-g` or `-l <value>`. It is recommended to use `-g`, since it's easier and you don't have to worry about the actual amount of memory the network might need.

Another kind of error is:
```
Internal error: plugin node (Unnamed Layer* 0) [PluginV2Ext]requires 20677888 bytes of scratch space, but only 20624360 is available. ...
../builder/cudnnBuilder2.cpp (1402) - OutOfMemory Error in buildSingleLayer: 0
```

This happens when the workspace memory limit for TensorRT is too low. It can be increased by `--wsz <value>` flag. The `<value>` is in bytes.

## Results of optimization
`YOLOv3` was tested on a computer with the following configuration:
```
Intel® Core™ i7-8850H CPU @ 2.60GHz × 12 
32GB RAM
Quadro P1000/PCIe/SSE2
1TB SSD

Ubuntu 18.04 x64
CUDA 10.1
cuBLAS 10
cuDNN 7
TensorRT 6
Python3
tensorflow 2.1
```

The results are as follows:
```
not-optimized, python3, average time per inference: ~220ms
optimized, python3, average time per inference: ~130ms\
not-optimized, C++ (Release), average time per inference: ~145ms
optimized, C++ (Release), average time per inference: 105ms
```

The final results may vary depending on hardware and how much a network can be optimized by TensorRT. Kepe in mind that some nodes can not be optimized by TensorRT.

## Further work
There might be some other optimization opportunities depending on the network. Also, the network should be tested with hardware that has support for FP16 or INT8.
