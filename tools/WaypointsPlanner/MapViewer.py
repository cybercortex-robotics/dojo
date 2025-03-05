"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing,
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""
import csv
import os
import time

import numpy as np
from PyQt5 import uic, QtWebEngineWidgets, QtWebChannel
from PyQt5.QtCore import Qt, pyqtSlot, QUrl
from PyQt5.QtGui import QStandardItemModel, QDoubleValidator
from PyQt5.QtWidgets import QWidget, QListView, QDoubleSpinBox, QPushButton, QFileDialog, QGridLayout, QMessageBox, \
    QLineEdit

from tools.WaypointsPlanner import conversion
from tools.WaypointsPlanner.GL.ListItem import ListItem
from tools.WaypointsPlanner.WaypointsPlanner import Landmark

"""
 * main_waypoints_planner_v3.py
 *
 *  Created on: 01.11.2023
 *      Author: Sorin Grigorescu
"""

current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "ui")


class MapViewer(QWidget):
    def __init__(self, main_window):
        super().__init__()
        uic.loadUi(os.path.join(ui_directory, 'MapViewer.ui'), self)
        self.setWindowTitle("CyberCortex.AI.Dojo Map Viewer V3")
        self.main_window = main_window

        # Widgets
        self.qtMapViewer = self.findChild(QWidget, 'qtMapViewer')
        self.qtGPSMapListView = self.findChild(QListView, 'qtGPSMapListView')

        self.qtLatitudeLineEdit = self.findChild(QLineEdit, "qtLatitudeLineEdit")
        self.qtLongitudeLineEdit = self.findChild(QLineEdit, "qtLongitudeLineEdit")
        self.qtAltitudeLineEdit = self.findChild(QLineEdit, "qtAltitudeLineEdit")

        self.qtLatitudeLineEdit.setValidator(QDoubleValidator(self))
        self.qtLongitudeLineEdit.setValidator(QDoubleValidator(self))
        self.qtAltitudeLineEdit.setValidator(QDoubleValidator(self))

        self.qtDeleteGPSPointPushButton = self.findChild(QPushButton, 'qtDeleteGPSPointPushButton')
        self.qtSaveGPSDataPushButton = self.findChild(QPushButton, 'qtSaveGPSDataPushButton')
        self.qtConvertToLandmarkPushButton = self.findChild(QPushButton, 'qtConvertToLandmarkPushButton')

        # Members
        self.model_map_gps = QStandardItemModel()
        self.qtGPSMapListView.setModel(self.model_map_gps)
        self.selection_model_GPS = self.qtGPSMapListView.selectionModel()

        self.map = QtWebEngineWidgets.QWebEngineView()
        self.channel = QtWebChannel.QWebChannel()
        self.markers = dict()
        self.selected_item = -1

        self.initConnections()
        self.initShortcuts()
        self.initWindow()

    # Init Methods -----------------------------------------------------------------------------------------------------
    def initConnections(self):
        self.selection_model_GPS.selectionChanged.connect(self.on_gps_selectionChanged)

        self.qtDeleteGPSPointPushButton.clicked.connect(self.on_deleteGPSPointButton_clicked)
        self.qtSaveGPSDataPushButton.clicked.connect(self.on_saveGPSDataButton_clicked)
        self.qtConvertToLandmarkPushButton.clicked.connect(self.on_convertToLandmarkButton_clicked)

        self.qtGPSMapListView.doubleClicked.connect(self.on_GPSMapListView_doubleClick)

        self.qtLatitudeLineEdit.textChanged.connect(self.on_latitudeLineEdit_textChanged)
        self.qtLongitudeLineEdit.textChanged.connect(self.on_longitudeLineEdit_textChanged)
        self.qtAltitudeLineEdit.textChanged.connect(self.on_altitudeLineEdit_textChanged)

    def initShortcuts(self):
        # Tooltips
        self.qtDeleteGPSPointPushButton.setToolTip("Shortcut -> Delete")
        self.qtSaveGPSDataPushButton.setToolTip("Shortcut -> CTRL + S")
        self.qtConvertToLandmarkPushButton.setToolTip("Shortcut -> C")

        # Shortcuts
        self.qtDeleteGPSPointPushButton.setShortcut(Qt.Key_Delete)
        self.qtSaveGPSDataPushButton.setShortcut(Qt.CTRL + Qt.Key_S)
        self.qtConvertToLandmarkPushButton.setShortcut(Qt.Key_C)

    def initWindow(self):
        self.channel.registerObject("MapWindow", self)
        self.map.page().setWebChannel(self.channel)

        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "map.html")
        self.map.setUrl(QUrl.fromLocalFile(file))

        layout = QGridLayout()
        layout.addWidget(self.map)
        self.qtMapViewer.setLayout(layout)

    # Methods ----------------------------------------------------------------------------------------------------------
    def loadCSVFile(self):
        file_gps = "{}{}.csv".format(''.join(self.main_window.waypoints_filename.split('.')[:-1]), "_gps")

        if not os.path.isfile(file_gps):
            print("Waypoints file ( {} ) does not exists.".format(file_gps))
            return

        # Parse map file
        csv_reader = csv.DictReader(open(file_gps, "r"))

        for idx, row in enumerate(csv_reader):
            try:
                lat = float(row["lat"])
                lng = float(row["lng"])
                alt = float(row["alt"])

                # print(lat, lng)
                # TODO: SHOW MARKERS
                self.map.page().runJavaScript("map.add_marker_at(L.latLng({}, {}));".format(lat, lng))
                self.markers[idx] = [lat, lng, alt]
            except:
                # print(idx)
                pass

        self.displayGPSLocation()

    def displayGPSLocation(self):
        self.model_map_gps.clear()
        for index, marker in self.markers.items():
            string_display = 'GPS #{}: ({:.5f}, {:.5f}, {:.2f})'.format(index, marker[0], marker[1], marker[2])
            self.model_map_gps.appendRow([ListItem(string_display)])

    def refreshGPSLocation(self):
        row = 0
        for idx, marker in self.markers.items():
            item = self.model_map_gps.item(row)
            string_display = 'GPS #{}: ({:.5f}, {:.5f}, {:.2f})'.format(idx, marker[0], marker[1], marker[2])
            item.setText(string_display)
            row += 1

    def selectCoordinates(self):
        if self.selected_item != -1 and self.selected_item in self.markers.keys():
            self.qtLatitudeLineEdit.setText(str(self.markers[self.selected_item][0]))
            self.qtLongitudeLineEdit.setText(str(self.markers[self.selected_item][1]))
            self.qtAltitudeLineEdit.setText(str(self.markers[self.selected_item][2]))

    def convertCoordinatesToLandmarks(self):
        first_marker = self.markers[list(self.markers.keys())[0]]
        sx, sy, _, _ = conversion.from_latlon(first_marker[0], first_marker[1])
        landmark_pts = list()
        for idx, marker in self.markers.items():
            lx, ly, _, _ = conversion.from_latlon(marker[0], marker[1])
            landmark_pts.append([idx, lx - sx, ly - sy, marker[2] - first_marker[2]])

        return landmark_pts

    # Slots ------------------------------------------------------------------------------------------------------------
    def on_convertToLandmarkButton_clicked(self):
        if len(self.markers) > 0:
            ans = QMessageBox.question(self, "Confirmation", "Convert GPS points to landmarks?")
            if ans == QMessageBox.StandardButton.Yes:
                self.main_window.WaypointsPlanner.clearLandmarks()
                for lm in self.convertCoordinatesToLandmarks():
                    landmark = Landmark(lm[0], np.array(lm[1:], dtype=np.float), np.zeros((3,), dtype=np.float), [],
                                        5.0)
                    self.main_window.WaypointsPlanner.landmarks.append(landmark)

    def on_gps_selectionChanged(self):
        self.selected_item = -1
        if len(self.selection_model_GPS.selection().indexes()) != 0:
            item = self.selection_model_GPS.selection().indexes()[0]

            self.selected_item = list(self.markers.keys())[item.row()]
            self.selectCoordinates()

    def on_deleteGPSPointButton_clicked(self):
        if self.selected_item != -1 and self.selected_item in self.markers.keys():
            self.map.page().runJavaScript("delete_marker_at({}, {});".format(self.markers[self.selected_item][0],
                                                                             self.markers[self.selected_item][1]))
            del self.markers[self.selected_item]
            self.displayGPSLocation()

    def on_saveGPSDataButton_clicked(self):
        file_gps = "{}{}.csv".format(''.join(self.main_window.waypoints_filename.split('.')[:-1]), "_gps")
        csv_file = QFileDialog.getSaveFileName(self, "Select CSV File",
                                               file_gps, "CSV files (*.csv)")

        if csv_file is not None and csv_file[0] != "":
            self.main_window.waypoints_filename = csv_file[0]
            writer = csv.DictWriter(open(csv_file[0], "w+", newline=''),
                                    ["timestamp_start", "timestamp_stop", "sampling_time", "lat", "lng", "alt"])
            timestamp_start = int(time.time()) * 1000  # [ms]
            sampling_time = 1  # [ms]

            writer.writeheader()
            for idx, marker in self.markers.items():
                writer.writerow({
                    "timestamp_start": timestamp_start,
                    "timestamp_stop": timestamp_start + (sampling_time * 1000),
                    "sampling_time": sampling_time * 1000,
                    "lat": marker[0],
                    "lng": marker[1],
                    "alt": marker[2]
                })

    def on_GPSMapListView_doubleClick(self):
        if self.selected_item != -1:
            marker = self.markers[self.selected_item]
            self.panMap(marker[0], marker[1])

    def on_latitudeLineEdit_textChanged(self):
        if self.selected_item != -1 and self.selected_item in self.markers.keys():
            coordinates = self.markers[self.selected_item]
            new_coordinates = [float(self.qtLatitudeLineEdit.text()), coordinates[1], coordinates[2]]
            self.map.page().runJavaScript("move_marker_at({}, {}, {}, {});".format(coordinates[0],
                                                                                   coordinates[1],
                                                                                   new_coordinates[0],
                                                                                   new_coordinates[1]))
            self.markers[self.selected_item] = new_coordinates
            self.refreshGPSLocation()

    def on_longitudeLineEdit_textChanged(self):
        if self.selected_item != -1 and self.selected_item in self.markers.keys():
            coordinates = self.markers[self.selected_item]
            new_coordinates = [coordinates[0], float(self.qtLongitudeLineEdit.text()), coordinates[2]]
            self.map.page().runJavaScript("move_marker_at({}, {}, {}, {});".format(coordinates[0],
                                                                                   coordinates[1],
                                                                                   new_coordinates[0],
                                                                                   new_coordinates[1]))
            self.markers[self.selected_item] = new_coordinates
            self.refreshGPSLocation()

    def on_altitudeLineEdit_textChanged(self):
        if self.selected_item != -1 and self.selected_item in self.markers.keys():
            coordinates = self.markers[self.selected_item]
            new_coordinates = [coordinates[0], coordinates[1], float(self.qtAltitudeLineEdit.text())]
            self.markers[self.selected_item] = new_coordinates
            self.refreshGPSLocation()

    # Map slots
    @pyqtSlot(int, float, float)
    def on_add_marker(self, idx, lat, lng):
        self.markers[idx] = [lat, lng, float(self.qtAltitudeLineEdit.text())]
        self.displayGPSLocation()

    @pyqtSlot(int)
    def on_remove_marker(self, idx):
        del self.markers[idx]
        self.displayGPSLocation()

    @pyqtSlot(int, float, float)
    def on_update_marker(self, idx, lat, lng):
        alt = float(self.qtLongitudeLineEdit.text())
        if idx in self.markers.keys():
            alt = self.markers[idx][2]

        self.markers[idx] = [lat, lng, alt]
        self.displayGPSLocation()
