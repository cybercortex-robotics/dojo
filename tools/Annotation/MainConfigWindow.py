"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from PyQt5.QtWidgets import QWidget, QLineEdit, QPushButton, QGroupBox, QGridLayout, QFileDialog, QListView
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5 import uic

from data.database_validation import ReturnCodes, validate_datastream_folder

from global_config import *
import json
import os

CFG = cfg
current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "ui")

CONFIGURATION_DIRECTORY = os.path.join(CFG.BASE.PATH, r"CyberCortex.AI\inference\etc\calibration\nuscenes") \
    if os.path.exists(os.path.join(CFG.BASE.PATH, r"CyberCortex.AI\inference\etc\calibration\nuscenes")) else "./"
OBJECT_CLASSES_DIR = os.path.join(CFG.BASE.PATH, r"CyberCortex.AI\inference\etc\env") \
    if os.path.join(CFG.BASE.PATH, r"CyberCortex.AI\inference\etc\env") else "./"


class MainConfigurationWindow(QWidget):
    def __init__(self, main_widget):
        super().__init__()
        uic.loadUi(os.path.join(ui_directory, 'MainConfigurationWindow.ui'), self)
        self.setWindowTitle("CyberCortex.AI Viewer")
        self.parent = main_widget

        # Widget
        self.dataBasePathLineEdit = self.findChild(QLineEdit, 'dataBasePathLineEdit')
        self.searchDataBasePushButton = self.findChild(QPushButton, 'searchDataBasePushButton')
        self.calibrationFilesGroupBox = self.findChild(QGroupBox, 'calibrationFilesGroupBox')
        self.finishPushButton = self.findChild(QPushButton, 'finishPushButton')
        self.historyListView = self.findChild(QListView, "historyListView")

        # Members
        self.row_grid = 0
        self.cameras = []
        self.lidars = []
        self.object_classes = ""
        self.base_path = ""

        self.calibrationFileWidget = QWidget(self)
        self.gridLayout = QGridLayout(self.calibrationFileWidget)

        self.model = QStandardItemModel()
        for path in self.parent.valid_db_paths:
            item = QStandardItem(path)
            self.model.appendRow(item)
        self.historyListView.setModel(self.model)

        # Signals
        self.searchDataBasePushButton.clicked.connect(self.on_search_datastream_path)
        self.dataBasePathLineEdit.textChanged.connect(self.change_db_path)
        self.historyListView.clicked.connect(self.on_select_path)

        # Initialization
        self.init()

    # Initialization ---------------------------------------------------------------------------------------------------
    def init(self):
        self.calibrationFileWidget.setLayout(self.gridLayout)
        self.finishPushButton.setVisible(False)
        self.calibrationFilesGroupBox.setVisible(False)

    # Methods ----------------------------------------------------------------------------------------------------------
    def process_db_path(self):
        if self.dataBasePathLineEdit.text() == "":
            return

        if not os.path.exists(self.parent.config_file):
            self.parent.on_display_error(
                message="Config file not found! Check if exists {}!".format(self.parent.config_file),
                error_id=ReturnCodes.CONFIG_NOT_FOUND)
            return

        self.parent.on_display_info(message="Configuration file {} found!".format(self.parent.config_file),
                                    info_id=ReturnCodes.OK)
        valid = validate_datastream_folder(db_path=self.dataBasePathLineEdit.text(),
                                           config_json=json.load(open(self.parent.config_file, "r")))

        if valid != ReturnCodes.OK:
            self.parent.on_display_error(message=ReturnCodes.code2string(valid), error_id=valid)

            if valid != ReturnCodes.FILTER_TYPE_INVALID:
                return

        if self.dataBasePathLineEdit.text() not in self.parent.valid_db_paths:
            self.parent.valid_db_paths.append(self.dataBasePathLineEdit.text())
            item = QStandardItem(self.dataBasePathLineEdit.text())
            self.model.appendRow(item)

        self.base_path = self.dataBasePathLineEdit.text()
        self.parent.on_display_info(message="Processed configuration file {} successfully!"
                                    .format(self.parent.config_file),
                                    info_id=valid)
        self.finishPushButton.setVisible(True)

    def change_db_path(self):
        self.finishPushButton.setVisible(False)
        self.process_db_path()

    # Slots ------------------------------------------------------------------------------------------------------------
    def on_db_path_text_change(self):
        self.change_db_path()

    def on_search_datastream_path(self):
        root_path = self.dataBasePathLineEdit.text()
        if root_path == "":
            root_path = "./"
        else:
            root_path = os.path.join(root_path, "..")

        self.dataBasePathLineEdit.setText("")
        folder_path = QFileDialog.getExistingDirectory(self, "Open Base Datastream Directory", root_path,
                                                       QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        self.dataBasePathLineEdit.setText(folder_path)

    def on_select_path(self, index):
        items = self.historyListView.selectedIndexes()
        for i in items:
            self.dataBasePathLineEdit.setText(i.data())
