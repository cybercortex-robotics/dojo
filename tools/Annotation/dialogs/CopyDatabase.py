"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal
from PyQt5 import uic

from global_config import *
import pandas as pd
import threading
import shutil
import sys
import os

sys.path.insert(1, '../')
current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")
CFG = cfg


########################################################################################################################
# IN PROGRESS ##########################################################################################################
########################################################################################################################
class CopyDatabase(QDialog):
    signal_finish = pyqtSignal(str)
    signal_warning = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(ui_directory, "CopyDatabaseWindow.ui"), self)
        self.setWindowTitle("Copy Database Window")

        # Widget
        self.qtCopyAllCheckBox       = self.findChild(QCheckBox   , 'qtCopyAllCheckBox')
        self.qtDatabasePathComboBox  = self.findChild(QComboBox   , 'qtDatabasePathComboBox')
        self.qtDatabasePushButton    = self.findChild(QPushButton , 'qtDatabasePushButton')
        self.qtDatastreamsListLayout = self.findChild(QVBoxLayout , 'qtDatastreamsListLayout')
        self.qtDirectoryLineEdit     = self.findChild(QLineEdit   , 'qtDirectoryLineEdit')
        self.qtDirectoryPushButton   = self.findChild(QPushButton , 'qtDirectoryPushButton')
        self.qtNameLineEdit          = self.findChild(QLineEdit   , 'qtNameLineEdit')
        self.qtCopyPushButton        = self.findChild(QPushButton , 'qtCopyPushButton')
        self.qtTitleLabel            = self.findChild(QLabel      , 'qtTitleLabel')
        self.qtDatastreamsScrollArea = self.findChild(QScrollArea , 'qtDatastreamsScrollArea')
        self.qtProgressBar           = self.findChild(QProgressBar, 'qtProgressBar')

        # Signals
        self.qtDatabasePushButton.clicked.connect(self.on_search_database_path)
        self.qtDirectoryPushButton.clicked.connect(self.on_search_destination_directory)
        self.qtCopyPushButton.clicked.connect(self.on_copy_database)

        self.qtDatabasePathComboBox.currentIndexChanged.connect(self.on_db_source_changed)
        self.qtCopyAllCheckBox.stateChanged.connect(self.on_check_box_state_changed)

        self.signal_finish.connect(self.on_finish)
        self.signal_warning.connect(self.on_show_warning)

        # Members
        self.datastreams = {}
        self.dependencies = {}
        self.loading = False

        # Init
        self.qtDatastreamsScrollArea.setEnabled(False)
        self.qtProgressBar.setVisible(False)

    # SLOTS ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def on_show_warning(text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Warning")
        msg.setText(text)
        msg.exec()

    def on_finish(self, error):
        self.qtCopyPushButton.setEnabled(True)
        self.qtProgressBar.setVisible(False)

        if error:
            self.qtTitleLabel.setText(error)

    def on_check_box_state_changed(self, state):
        self.qtDatastreamsScrollArea.setEnabled(not state)

    def on_db_source_changed(self, index):
        self.set_paths(self.qtDatabasePathComboBox.itemText(index))

        for i in reversed(range(self.qtDatastreamsListLayout.count())):
            self.qtDatastreamsListLayout.itemAt(i).widget().deleteLater()

        self.datastreams.clear()
        self.dependencies.clear()
        self.fill_scroll_area()

    def on_search_database_path(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Base Datastream Directory",
                                                       self.qtDirectoryLineEdit.text(),
                                                       QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        print("on_search_database_path", folder_path)
        self.set_paths(folder_path)

    def on_search_destination_directory(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Base Datastream Directory",
                                                       self.qtDirectoryLineEdit.text(),
                                                       QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        print("on_search_destination_directory", folder_path)
        self.qtDirectoryLineEdit.setText(folder_path)

    def on_copy_database(self):
        self.qtCopyPushButton.setEnabled(False)
        self.qtProgressBar.setVisible(True)

        threading.Thread(target=self.copy_database).start()

    # Methods ----------------------------------------------------------------------------------------------------------
    def copy_database(self):
        source_db = self.qtDatabasePathComboBox.currentText()
        destination_db = os.path.join(self.qtDirectoryLineEdit.text(), self.qtNameLineEdit.text())

        if os.path.exists(destination_db):
            self.signal_finish.emit("Copy failed! Destination file exists! {}".format(destination_db))
            return

        self.qtTitleLabel.setText("Copy {} to {}".format(source_db, destination_db))

        if self.qtCopyAllCheckBox.isChecked():
            self.copy_whole_database(source_db, destination_db)
        else:
            if not self.check_selected_datastreams():
                self.signal_finish.emit("Copy failed!")
                return

            datastream_keys = [key for key in self.datastreams if self.datastreams[key][0].isChecked()]
            self.copy_partial_database(source_db, destination_db, datastream_keys)

        self.signal_finish.emit("Copy {} to {} finished!".format(source_db, destination_db))

    def copy_whole_database(self, source_db, destination_db):
        try:
            shutil.copytree(source_db, destination_db)
        except Exception as e:
            self.signal_finish.emit("Copy failed! {}".format(e))

    def copy_partial_database(self, source_db, destination_db, datastream_keys):
        # DIRECTORIES --------------------------------------------------------------------------------------------------
        nr_directories = len(datastream_keys)
        for i in range(nr_directories):
            key = datastream_keys[i]
            datastream_source_db = os.path.join(source_db, "datastream_{}".format(key))
            datastream_destination_db = os.path.join(destination_db, "datastream_{}".format(key))

            try:
                shutil.copytree(datastream_source_db, datastream_destination_db)
            except Exception as e:
                self.signal_finish.emit("Copy datastream {} failed! {}".format(key, e))

        # datablock_descriptor -----------------------------------------------------------------------------------------
        source_datablock_descriptor_path = os.path.join(source_db, "datablock_descriptor.csv")
        destination_datablock_descriptor_path = os.path.join(destination_db, "datablock_descriptor.csv")

        try:
            with open(source_datablock_descriptor_path, "r") as datablock_descriptor_source:
                with open(destination_datablock_descriptor_path, "w") as datablock_descriptor_destination:
                    lines = datablock_descriptor_source.readlines()
                    datablock_descriptor_destination.write(lines[0])

                    for line in lines[1:]:
                        if line.split(',')[1] in datastream_keys:
                            datablock_descriptor_destination.write(line)
        except Exception as e:
            self.signal_finish.emit("Copy datablock_descriptor failed! {}".format(e))

        # sampling_timestamps_sync -------------------------------------------------------------------------------------
        source_sampling_timestamps_path = os.path.join(source_db, "sampling_timestamps_sync.csv")
        destination_sampling_timestamps_path = os.path.join(destination_db, "sampling_timestamps_sync.csv")

        try:
            datastream_keys = ["datastream_{}".format(key) for key in datastream_keys]
            datastream_keys.insert(0, 'timestamp_stop')

            data = pd.read_csv(source_sampling_timestamps_path)
            data = data[datastream_keys]
            data.to_csv(destination_sampling_timestamps_path, index=False)
        except Exception as e:
            self.signal_finish.emit("Copy datablock_descriptor failed! {}".format(e))

    def check_selected_datastreams(self):
        for key, dependencies in self.dependencies.items():
            if not self.datastreams[key][0].isChecked():
                continue

            for dependency in dependencies:
                if not self.datastreams[dependency][0].isChecked():
                    self.signal_warning.emit("Datastream {} is dependent on {}! Please check {} also!"
                                             .format(key, dependency, dependency))
                    return False

        return True

    def fill_scroll_area(self):
        file_path = os.path.join(self.qtDatabasePathComboBox.currentText(), "datablock_descriptor.csv")

        with open(file_path, "r") as datablock_descriptor:
            for datastream in datablock_descriptor.readlines()[1:]:
                datastream_info = datastream.strip().split(',')

                qt_check_box = QCheckBox("{}: {}".format(datastream_info[1], datastream_info[2]))
                qt_check_box.setChecked(False)

                self.qtDatastreamsListLayout.addWidget(qt_check_box)
                self.datastreams[datastream_info[1]] = [qt_check_box, datastream_info[1:]]

                if datastream_info[-1] == '':
                    continue

                dependencies = datastream_info[-1][1:-1].split(';')
                self.dependencies[datastream_info[1]] = []
                for dependency in dependencies:
                    self.dependencies[datastream_info[1]].append(dependency.split('-')[-1])

    def set_paths(self, db_source_path):
        if not os.path.exists(db_source_path):
            return

        if db_source_path not in [self.qtDatabasePathComboBox.itemText(i)
                                  for i in range(self.qtDatabasePathComboBox.count())]:
            if db_source_path[-1] != '/':
                db_source_path += '/'
            self.qtDatabasePathComboBox.addItem(db_source_path)

        self.qtDatabasePathComboBox.setCurrentText(db_source_path)
        self.qtDirectoryLineEdit.setText(os.path.split(os.path.dirname(db_source_path))[0])
        self.qtNameLineEdit.setText("{}-copy".format(os.path.basename(os.path.dirname(db_source_path))))

    def prepare(self, history_paths):
        self.set_paths(CFG.DB.BASE_PATH)

        for db_path in history_paths:
            if db_path[-1] != '/':
                db_path += '/'
            self.qtDatabasePathComboBox.addItem(db_path)


def main():
    try:
        app = QApplication([])
        window = CopyDatabase()
        window.show()
        app.exec_()
    except RuntimeError:
        exit(0)


if __name__ == "__main__":
    main()
