"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QApplication, QTextEdit, QSplitter, QMessageBox
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt
from PyQt5 import uic

from data.database_validation import validate_datastream_folder, ReturnCodes
from tools.Annotation.MainConfigWindow import MainConfigurationWindow
from tools.Annotation.MainControlWindow import MainControlWindow

import global_config
import json
import sys
import os

sys.path.insert(1, '../../')
try:
    sys.path.append(os.environ['CyC_DIR'] + r'/dojo')
except:
    print("CyC_DIR is undefined. Using default path \"C:/dev/src/CyberCortex.AI\".")


CFG = global_config.cfg
current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "ui")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(ui_directory, 'MainWindow.ui'), self)
        self.setWindowTitle("CyberCortex.AI.Dojo Annotation")

        # Widget
        self.qtMainStackedWidget = self.findChild(QStackedWidget, 'qtMainStackedWidget')
        self.logsScrollArea      = self.findChild(QTextEdit, 'logsScrollArea')
        self.splitter            = self.findChild(QSplitter, 'splitter')

        # Config
        self.config_file = "config.json"

        self.valid_db_paths = []
        if os.path.exists(self.config_file):
            self.valid_db_paths = self._load_history_items()

        # Members
        self.qtMainConfiguration = MainConfigurationWindow(main_widget=self)
        self.qtMainControl = None

        # Configuration Signals
        self.qtMainConfiguration.finishPushButton.clicked.connect(self.on_finish_configuration)

        # Initialization
        self._init_stacked_widget()
        self.logsScrollArea.setReadOnly(True)
        self.splitter.setSizes([300, 25])

        self.hide()

    # Events -----------------------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        result = QMessageBox.question(self,
                                      "Confirm Exit...",
                                      "Are you sure you want to exit ?",
                                      QMessageBox.Yes | QMessageBox.No)
        event.ignore()
        if result == QMessageBox.Yes:
            if self.qtMainControl is not None:
                pass
                # self.qtMainControl.qtViewingArea.currentWidget().clear_selection()
            event.accept()

    # Initialization ---------------------------------------------------------------------------------------------------
    def _init_stacked_widget(self):
        self.qtMainStackedWidget.addWidget(self.qtMainConfiguration)
        self.qtMainStackedWidget.setCurrentWidget(self.qtMainConfiguration)

    # Private Methods --------------------------------------------------------------------------------------------------
    def _load_history_items(self):
        """
        Load list of recent databases. Discard invalid paths.
        """
        with open(self.config_file, "r") as json_conf:
            j_conf = json.load(json_conf)

            if "HistoryDatabases" not in j_conf:
                j_conf["HistoryDatabases"] = []
            else:
                valid_db_path = []
                for idx in range(len(j_conf["HistoryDatabases"])):
                    return_code = validate_datastream_folder(j_conf["HistoryDatabases"][idx], j_conf)
                    if return_code == ReturnCodes.OK or return_code == ReturnCodes.FILTER_TYPE_INVALID:
                        valid_db_path.append(j_conf["HistoryDatabases"][idx])
        return valid_db_path

    def _update_recent_database_folders(self):
        with open(self.config_file, "r") as json_conf:
            json_obj = json.load(json_conf)

        self.valid_db_paths = self.valid_db_paths[-5:]
        json_obj["HistoryDatabases"] = self.valid_db_paths

        with open(self.config_file, "w") as json_conf:
            json.dump(json_obj, json_conf, indent=4)

    # SLOTS Messages ---------------------------------------------------------------------------------------------------
    def on_display_info(self, message, info_id=0):
        self.logsScrollArea.setTextColor(Qt.black)
        self.logsScrollArea.append("Info FILTER {}: {}".format(info_id, message))
        self.logsScrollArea.textCursor().movePosition(QTextCursor.End)

    def on_display_warning(self, message, warning_id=0):
        self.logsScrollArea.setTextColor(Qt.darkYellow)
        self.logsScrollArea.append("Warning FILTER {}: {}".format(warning_id, message))
        self.logsScrollArea.textCursor().movePosition(QTextCursor.End)

    def on_display_error(self, message, error_id=0):
        self.logsScrollArea.setTextColor(Qt.red)
        self.logsScrollArea.append("ERROR FILTER {}: {}".format(error_id, message))
        self.logsScrollArea.textCursor().movePosition(QTextCursor.End)

    # SLOTS ------------------------------------------------------------------------------------------------------------
    def on_back_to_configuration_window(self):
        self.qtMainStackedWidget.setCurrentWidget(self.qtMainConfiguration)

    def on_finish_configuration(self):
        CFG.DB.BASE_PATH = self.qtMainConfiguration.base_path

        if self.qtMainControl is None:
            self.qtMainControl = MainControlWindow(main_widget=self)
            self.qtMainStackedWidget.addWidget(self.qtMainControl)
            self.qtMainStackedWidget.setCurrentWidget(self.qtMainControl)
        else:
            self.qtMainStackedWidget.setCurrentWidget(self.qtMainControl)
            self.qtMainControl.reload_db()

        self._update_recent_database_folders()
        self.qtMainControl.show_first_data()


def main():
    try:
        app = QApplication([])
        window = MainWindow()
        window.show()
        app.exec_()
    except RuntimeError:
        exit(0)


if __name__ == "__main__":
    main()
