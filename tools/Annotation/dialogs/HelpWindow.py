"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QLabel, QTextEdit, QApplication
from PyQt5 import uic

from global_config import *
import sys

sys.path.insert(1, '../')
current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")
CFG = cfg


class HelpWindow(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(ui_directory, "HelpWindow.ui"), self)
        self.setWindowTitle("Help Window")

        # Widget
        self.qtTitle = self.findChild(QLabel, "qtTitle")
        self.qtText = self.findChild(QTextEdit, "qtText")

        self.path = os.path.join(current_directory, "help_text/temp_help.txt")

    # Events -----------------------------------------------------------------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()

        ctrl = False
        if event.modifiers() and Qt.ControlModifier:
            ctrl = True

        if ctrl and key == Qt.Key_S:
            self.save_file()
        super().keyPressEvent(event)

    # Initialization Methods -------------------------------------------------------------------------------------------
    def load_file(self, path, filter_name):
        self.qtTitle.setText(filter_name)
        self.path = path

        with open(self.path, "r") as file:
            content = file.read()
            self.qtText.setTextColor(Qt.black)
            self.qtText.setText(content)

    def save_file(self):
        with open(self.path, 'w') as file:
            file.write(str(self.qtText.toPlainText()))


def main():
    try:
        app = QApplication([])
        window = HelpWindow()
        window.show()
        app.exec_()
    except RuntimeError:
        exit(0)


if __name__ == "__main__":
    main()
