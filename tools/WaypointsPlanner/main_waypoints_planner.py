"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import os
import sys

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QApplication, QWidget, QGridLayout, QAction, QFileDialog

# Append CyberCortex.AI dojo to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.WaypointsPlanner.MapViewer import MapViewer
from tools.WaypointsPlanner.LandmarksViewer import LandmarksViewer
from tools.WaypointsPlanner.WaypointsPlanner import WaypointsPlanner

"""
 * main_waypoints_planner.py
 *
 *  Created on: 01.11.2023
 *      Author: Sorin Grigorescu
"""

current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "ui")


class WaypointsPlannerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(ui_directory, 'MainWaypointsPlannerWindow.ui'), self)
        self.setWindowTitle("CyberCortex.AI.Dojo Waypoints Planner")

        # Widget
        self.tabWidget = self.findChild(QTabWidget, 'tabWidget')
        self.qtLandmarksWidget = self.findChild(QWidget, 'qtLandmarksWidget')
        self.qtMapWidget = self.findChild(QWidget, 'qtMapWidget')

        # Actions
        self.qtLoadWaypointsAction = self.findChild(QAction, "qtLoadWaypointsAction")
        self.qtLoadMapsAction = self.findChild(QAction, "qtLoadMapsAction")

        # Tooltips
        self.qtLoadWaypointsAction.setToolTip("Shortcut -> CTRL + L")

        # Shortcuts
        self.qtLoadWaypointsAction.setShortcut(Qt.CTRL + Qt.Key_W)
        self.qtLoadMapsAction.setShortcut(Qt.CTRL + Qt.Key_M)

        # Signals
        self.qtLoadWaypointsAction.triggered.connect(self.on_loadWaypoints_triggeredAction)
        self.qtLoadMapsAction.triggered.connect(self.on_loadMaps_triggeredAction)
        self.tabWidget.currentChanged.connect(self.on_tab_changed)

        # Initialization
        self.WaypointsPlanner = WaypointsPlanner()

        self.qtLandmarksViewer = LandmarksViewer(self)
        self.qtMapViewer = MapViewer(self)

        self.history_waypoints_path = "temp_waypoints_history.conf"
        self.waypoints_filename = self.readLastPath(self.history_waypoints_path)

        self.history_map_path = "temp_map_history.conf"
        self.map_filename = self.readLastPath(self.history_map_path)

        self.loadViewers(self.qtLandmarksWidget, self.qtLandmarksViewer)
        self.loadViewers(self.qtMapWidget, self.qtMapViewer)

        if os.path.exists(self.waypoints_filename):
            self.setWindowTitle("CyberCortex.AI.Dojo Waypoints Planner - {}".format(self.waypoints_filename))
            self.qtLandmarksViewer.loadWaypointsFile()
            self.qtMapViewer.loadCSVFile()
        else:
            print("No waypoints filename was found!")

        #if os.path.exists(self.map_filename):
        #    self.qtLandmarksViewer.loadMapFile()
        #else:
        #    print("No map filename was found!")

    # Events -----------------------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        self.writeLastPath(self.history_waypoints_path, self.waypoints_filename)
        self.writeLastPath(self.history_map_path, self.map_filename)

    # Signals ----------------------------------------------------------------------------------------------------------
    def on_loadWaypoints_triggeredAction(self):
        csv_file = QFileDialog.getOpenFileName(self, "Select CSV file", self.waypoints_filename, "CSV files (*.csv)")
        if csv_file is not None and csv_file[0] != "":
            self.waypoints_filename = csv_file[0]
            self.setWindowTitle("CyberCortex.AI.Dojo Waypoints Planner - {}".format(self.waypoints_filename))
            self.qtLandmarksViewer.loadWaypointsFile()
            self.qtMapViewer.loadCSVFile()

    def on_loadMaps_triggeredAction(self):
        map_file = QFileDialog.getOpenFileName(self, "Select MAP file", self.map_filename, "MAP files (*.map)")
        if map_file is not None and map_file[0] != "":
            self.map_filename = map_file[0]
            self.setWindowTitle("CyberCortex.AI.Dojo Waypoints Planner - {}".format(self.map_filename))
            self.qtLandmarksViewer.loadMapFile()

    def on_tab_changed(self):
        self.qtLandmarksViewer.refresh()

    # Static Methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def loadViewers(widget, viewer):
        layout = QGridLayout()
        layout.addWidget(viewer)
        widget.setLayout(layout)

    # Static Methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def readLastPath(path):
        if os.path.exists(path):
            history_file = open(path, "r")
            csv_path = history_file.readline()
            history_file.close()

            if len(csv_path):
                return csv_path

        return "./"

    @staticmethod
    def writeLastPath(file, path):
        history_file = open(file, "w")
        if path:
            history_file.write(path)
        history_file.close()


def main():
    try:
        app = QApplication(sys.argv)
        window = WaypointsPlannerMainWindow()
        window.show()
        app.exec()
    except RuntimeError:
        exit(0)


if __name__ == "__main__":
    main()
