"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import cv2
import os

db_path = r'C:\data\Carla\SemSeg_Town03_original'
rgb_path = r'datastream_1\samples\0\left\1676906910.png'
rgb_path = os.path.join(db_path, rgb_path)
semseg_path = r'datastream_7\samples\0\left\1676906910.png'
semseg_path = os.path.join(db_path, semseg_path)

# Load the image
img = cv2.imread(rgb_path)
semseg = cv2.imread(semseg_path)

# Create a window
cv2.namedWindow('image')

# Mouse callback function
def color_picker(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the BGR values of the clicked pixel
        b, g, r = semseg[y, x]
        print('BGR: ({}, {}, {})'.format(b, g, r))

        # Display the image
        cv2.imshow('image', img)

# Set the mouse callback function
cv2.setMouseCallback('image', color_picker)

# Display the image
cv2.imshow('image', img)

# Wait for a key press
while cv2.waitKey(0) != ord('q'):
    pass

# Destroy all windows
cv2.destroyAllWindows()
