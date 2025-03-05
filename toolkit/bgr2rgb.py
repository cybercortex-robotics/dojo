import numpy as np
import os
import cv2

folder = r"C:\data\UnitreeA1\unitree_indoor_icdt_01\datastream_1\samples\0\left"

# Parse annotation images in the folder
annotations = []
for file in os.listdir(folder):
    if file.endswith(".png"):
        annotations.append(os.path.join(folder, file))

# Display the annotation images
for annotation in annotations:
    img = cv2.imread(annotation, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # cv2.imwrite(annotation, img)

    cv2.imshow("Annotation", img)
    cv2.waitKey(1)
