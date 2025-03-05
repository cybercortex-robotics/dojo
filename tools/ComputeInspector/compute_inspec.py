"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import psutil

# gives a single float value
#print("CPU:", psutil.cpu_percent())
# gives an object with many fields
#print(psutil.virtual_memory())
# you can convert that object to a dictionary
#dict(psutil.virtual_memory()._asdict())
# you can have the percentage of used RAM
#print(psutil.virtual_memory().percent)
# you can calculate percentage of available memory
#print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)