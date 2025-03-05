## PyCocoTools

This is used to decode and generate stuff segmentation.

# Source
https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools

# Setup
The important component it's written in Cython (`_mask.pyx`).

In the repo you will find the 2 files that are needed for this: `_mask.cp37-win_amd64.pyd` 
and `_mask.c`. If they do not work for you, follow the Build Tools section.

# Build Tools
 - Download [cocoapi repo](https://github.com/cocodataset/cocoapi)
 - Run `<root>/PythonAPI/setup.py` with the following line arguments: `build_ext --inplace`
 - Inside `<root>/PythonAPI/pycocotools` you will find the 2 files needed for this. Copy them into `converters/coco/pycocotools`
 - You are good to go