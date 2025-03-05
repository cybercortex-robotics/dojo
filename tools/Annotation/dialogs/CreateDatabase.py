"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5 import uic
from global_config import *
import sys

sys.path.insert(1, '../')
current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")
CFG = cfg


########################################################################################################################
# IN PROGRESS ##########################################################################################################
########################################################################################################################
class CreateDatabase(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(ui_directory, "CreateDatabaseWindow.ui"), self)
        self.setWindowTitle("Create Database Window -- IN PROGRESS")


def main():
    try:
        app = QApplication([])
        window = CreateDatabase()
        window.show()
        app.exec_()
    except RuntimeError:
        exit(0)


if __name__ == "__main__":
    main()
