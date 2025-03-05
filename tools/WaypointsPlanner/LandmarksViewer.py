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

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5 import uic
from PyQt5.QtCore import Qt, QItemSelectionModel
from PyQt5.QtGui import QVector3D, QStandardItemModel, QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QSlider, QCheckBox, \
    QListView, QFileDialog, QMenu, QLabel, QLineEdit

from tools.WaypointsPlanner.GL.GLAxisItemWaypointsPlanner import GLAxisItemWaypointsPlanner
from tools.WaypointsPlanner.GL.GLCubeItemWaypointsPlanner import GLCubeItemWaypointsPlanner
from tools.WaypointsPlanner.GL.GLViewWidgetWaypointsPlanner import GLViewWidgetWaypointsPlanner
from tools.WaypointsPlanner.GL.ListItem import DISTANCE, AXIS_LENGTH, ListItem, LINE_WIDTH, NODE_WIDTH
from tools.WaypointsPlanner.WaypointsPlanner import Waypoint, Landmark

from toolkit.env.CMap import CMap
from toolkit.env.CMapStorage import CMapStorage

"""
 * main_waypoints_planner_v3.py
 *
 *  Created on: 01.11.2023
 *      Author: Sorin Grigorescu
"""

current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "ui")


class LandmarksViewer(QWidget):
    def __init__(self, main_window):
        super().__init__()
        uic.loadUi(os.path.join(ui_directory, 'LandmarksViewer.ui'), self)
        self.setWindowTitle("CyberCortex.AI.Dojo Landmarks Viewer V3")
        self.main_window = main_window

        # Widgets
        self.qtPoseLabel = self.findChild(QLabel, 'qtPoseLabel')
        self.qtGlViewer = self.findChild(QWidget, 'qtGlViewer')
        self.qtLandmarksListView = self.findChild(QListView, 'qtLandmarksListView')

        # Push Buttons
        self.qtAddLandmarkPushButton = self.findChild(QPushButton, 'qtAddLandmarkPushButton')
        self.qtAddWaypointPushButton = self.findChild(QPushButton, 'qtAddWaypointPushButton')
        self.qtDeleteWaypointPushButton = self.findChild(QPushButton, 'qtDeleteWaypointPushButton')
        self.qtMoveUpPushButton = self.findChild(QPushButton, "qtMoveUpPushButton")
        self.qtMoveDownPushButton = self.findChild(QPushButton, "qtMoveDownPushButton")
        self.qtSaveToCsvPushButton = self.findChild(QPushButton, 'qtSaveToCsvPushButton')

        # Check Boxes
        self.qtQuaternionCheckBox = self.findChild(QCheckBox, 'qtQuaternionCheckBox')
        self.qtTopViewCheckBox = self.findChild(QCheckBox, 'qtTopViewCheckBox')
        self.qtFrontViewCheckBox = self.findChild(QCheckBox, 'qtFrontViewCheckBox')
        self.qtSideViewCheckBox = self.findChild(QCheckBox, 'qtSideViewCheckBox')

        # Line Edits
        self.qtCoordsMinLineEdit = self.findChild(QLineEdit, "qtCoordsMinLineEdit")
        self.qtCoordsMaxLineEdit = self.findChild(QLineEdit, "qtCoordsMaxLineEdit")
        self.qtCoordsStepLineEdit = self.findChild(QLineEdit, "qtCoordsStepLineEdit")
        self.qtAngleStepLineEdit = self.findChild(QLineEdit, "qtAngleStepLineEdit")

        self.qtXLineEdit = self.findChild(QLineEdit, "qtXLineEdit")
        self.qtYLineEdit = self.findChild(QLineEdit, "qtYLineEdit")
        self.qtZLineEdit = self.findChild(QLineEdit, "qtZLineEdit")

        self.qtRollLineEdit = self.findChild(QLineEdit, "qtRollLineEdit")
        self.qtPitchLineEdit = self.findChild(QLineEdit, "qtPitchLineEdit")
        self.qtYawLineEdit = self.findChild(QLineEdit, "qtYawLineEdit")

        self.qtTravelTimeLineEdit = self.findChild(QLineEdit, "qtTravelTimeLineEdit")
        self.qtLandmarkIDLineEdit = self.findChild(QLineEdit, "qtLandmarkIDLineEdit")

        self.qtCoordsMinLineEdit.setValidator(QDoubleValidator(self))
        self.qtCoordsMaxLineEdit.setValidator(QDoubleValidator(self))
        self.qtCoordsStepLineEdit.setValidator(QDoubleValidator(self))
        self.qtAngleStepLineEdit.setValidator(QDoubleValidator(self))

        self.qtXLineEdit.setValidator(QDoubleValidator(self))
        self.qtYLineEdit.setValidator(QDoubleValidator(self))
        self.qtZLineEdit.setValidator(QDoubleValidator(self))

        self.qtRollLineEdit.setValidator(QDoubleValidator(self))
        self.qtPitchLineEdit.setValidator(QDoubleValidator(self))
        self.qtYawLineEdit.setValidator(QDoubleValidator(self))

        self.qtTravelTimeLineEdit.setValidator(QDoubleValidator(self))
        self.qtLandmarkIDLineEdit.setValidator(QIntValidator(self))

        # Sliders
        self.qtXSlider = self.findChild(QSlider, "qtXSlider")
        self.qtYSlider = self.findChild(QSlider, "qtYSlider")
        self.qtZSlider = self.findChild(QSlider, "qtZSlider")

        self.qtRollSlider = self.findChild(QSlider, "qtRollSlider")
        self.qtPitchSlider = self.findChild(QSlider, "qtPitchSlider")
        self.qtYawSlider = self.findChild(QSlider, "qtYawSlider")

        # Members
        self.model_landmarks = QStandardItemModel()
        self.map = CMap()
        self.qtLandmarksListView.setModel(self.model_landmarks)
        self.landmarks_selection_model = self.qtLandmarksListView.selectionModel()

        self.gl_view = GLViewWidgetWaypointsPlanner()

        self.selected_item = [-1, -1]
        self.original_position = list()
        self.mouse_press_on_axis = -1
        self.mouse_is_down = False
        self.prev_camera_position = None

        # Initialisation
        self.initShortcuts()
        self.initConnections()
        self.initWindow()
        self.initScene()
        self.initTabOrder()

    # Init Methods -----------------------------------------------------------------------------------------------------
    def initTabOrder(self):
        QWidget.setTabOrder(self.qtCoordsMinLineEdit, self.qtCoordsMaxLineEdit)
        QWidget.setTabOrder(self.qtCoordsMaxLineEdit, self.qtCoordsStepLineEdit)
        QWidget.setTabOrder(self.qtCoordsStepLineEdit, self.qtAngleStepLineEdit)
        QWidget.setTabOrder(self.qtAngleStepLineEdit, self.qtXSlider)
        QWidget.setTabOrder(self.qtXSlider, self.qtXLineEdit)
        QWidget.setTabOrder(self.qtXLineEdit, self.qtYSlider)
        QWidget.setTabOrder(self.qtYSlider, self.qtYLineEdit)
        QWidget.setTabOrder(self.qtYLineEdit, self.qtZSlider)
        QWidget.setTabOrder(self.qtZSlider, self.qtZLineEdit)
        QWidget.setTabOrder(self.qtZLineEdit, self.qtRollSlider)
        QWidget.setTabOrder(self.qtRollSlider, self.qtRollLineEdit)
        QWidget.setTabOrder(self.qtRollLineEdit, self.qtPitchSlider)
        QWidget.setTabOrder(self.qtPitchSlider, self.qtPitchLineEdit)
        QWidget.setTabOrder(self.qtPitchLineEdit, self.qtYawSlider)
        QWidget.setTabOrder(self.qtYawSlider, self.qtYawLineEdit)
        QWidget.setTabOrder(self.qtYawLineEdit, self.qtLandmarkIDLineEdit)
        QWidget.setTabOrder(self.qtLandmarkIDLineEdit, self.qtTravelTimeLineEdit)
        QWidget.setTabOrder(self.qtTravelTimeLineEdit, self.qtMoveUpPushButton)
        QWidget.setTabOrder(self.qtMoveUpPushButton, self.qtMoveDownPushButton)
        QWidget.setTabOrder(self.qtMoveDownPushButton, self.qtSaveToCsvPushButton)
        QWidget.setTabOrder(self.qtSaveToCsvPushButton, self.qtQuaternionCheckBox)

    def initConnections(self):
        # Viewer
        self.gl_view.signal_mouse_press.connect(self.on_gl_view_mouse_press)
        self.gl_view.signal_mouse_release.connect(self.on_gl_view_mouse_release)
        self.gl_view.signal_mouse_move.connect(self.on_gl_view_mouse_move)

        # List View
        self.landmarks_selection_model.selectionChanged.connect(self.on_landmarkList_selectionChanged)
        self.qtLandmarksListView.clicked.connect(self.on_landmarkList_clicked)
        self.qtLandmarksListView.customContextMenuRequested.connect(self.on_landmarkList_customContextMenuRequested)

        # Check Boxes
        self.qtTopViewCheckBox.stateChanged.connect(self.on_topViewCheckBox_stateChanged)
        self.qtFrontViewCheckBox.stateChanged.connect(self.on_frontViewCheckBox_stateChanged)
        self.qtSideViewCheckBox.stateChanged.connect(self.on_sideViewCheckBox_stateChanged)

        # Double Spin Boxes
        self.qtCoordsMinLineEdit.textChanged.connect(self.on_coordinatesMinLineEdit_textChanged)
        self.qtCoordsMaxLineEdit.textChanged.connect(self.on_coordinatesMaxLineEdit_textChanged)
        self.qtCoordsStepLineEdit.textChanged.connect(self.on_coordinatesStepLineEdit_textChanged)
        self.qtAngleStepLineEdit.textChanged.connect(self.on_angleStepLineEdit_textChanged)

        self.qtXLineEdit.textChanged.connect(self.on_xLineEdit_textChanged)
        self.qtYLineEdit.textChanged.connect(self.on_yLineEdit_textChanged)
        self.qtZLineEdit.textChanged.connect(self.on_zLineEdit_textChanged)

        self.qtRollLineEdit.textChanged.connect(self.on_rollLineEdit_textChanged)
        self.qtPitchLineEdit.textChanged.connect(self.on_pitchLineEdit_textChanged)
        self.qtYawLineEdit.textChanged.connect(self.on_yawLineEdit_textChanged)

        # Slides
        self.qtXSlider.valueChanged.connect(self.on_xSlider_valueChanged)
        self.qtYSlider.valueChanged.connect(self.on_ySlider_valueChanged)
        self.qtZSlider.valueChanged.connect(self.on_zSlider_valueChanged)

        self.qtRollSlider.valueChanged.connect(self.on_rollSlider_valueChanged)
        self.qtPitchSlider.valueChanged.connect(self.on_pitchSlider_valueChanged)
        self.qtYawSlider.valueChanged.connect(self.on_yawSlider_valueChanged)

        # Push Buttons
        self.qtAddLandmarkPushButton.clicked.connect(self.on_addLandmarkButton_clicked)
        self.qtAddWaypointPushButton.clicked.connect(self.on_addWaypointButton_clicked)
        self.qtDeleteWaypointPushButton.clicked.connect(self.on_deleteWaypointButton_clicked)
        self.qtMoveUpPushButton.clicked.connect(self.on_MoveUpButton_clicked)
        self.qtMoveDownPushButton.clicked.connect(self.on_MoveDownButton_clicked)
        self.qtSaveToCsvPushButton.clicked.connect(self.on_saveToCSVButton_clicked)

    def initShortcuts(self):
        # Tooltips
        self.qtAddLandmarkPushButton.setToolTip("Shortcut -> SHIFT + L")
        self.qtAddWaypointPushButton.setToolTip("Shortcut -> SHIFT + W")
        self.qtDeleteWaypointPushButton.setToolTip("Shortcut -> Delete")
        self.qtMoveUpPushButton.setToolTip("Shortcut -> SHIFT + 8")
        self.qtMoveDownPushButton.setToolTip("Shortcut -> SHIFT + 9")
        self.qtSaveToCsvPushButton.setToolTip("Shortcut -> CTRL + S")

        # Shortcuts
        self.qtAddLandmarkPushButton.setShortcut(Qt.SHIFT + Qt.Key_L)
        self.qtAddWaypointPushButton.setShortcut(Qt.SHIFT + Qt.Key_W)
        self.qtDeleteWaypointPushButton.setShortcut(Qt.Key_Delete)
        self.qtMoveUpPushButton.setShortcut(Qt.SHIFT + Qt.Key_8)
        self.qtMoveDownPushButton.setShortcut(Qt.SHIFT + Qt.Key_9)
        self.qtSaveToCsvPushButton.setShortcut(Qt.CTRL + Qt.Key_S)

    def initWindow(self):
        self.gl_view.setCameraPosition(distance=DISTANCE)

        layout = QGridLayout()
        self.qtGlViewer.setLayout(layout)
        self.qtGlViewer.layout().addWidget(self.gl_view)

        self.on_coordinatesStepLineEdit_textChanged()
        self.on_angleStepLineEdit_textChanged()

    def initScene(self):
        z_grid = gl.GLGridItem()
        z_grid.scale(AXIS_LENGTH, AXIS_LENGTH, AXIS_LENGTH)
        self.gl_view.addItem(z_grid)

        # Add (0, 0, 0) coordinate axes
        axis = GLAxisItemWaypointsPlanner()
        self.gl_view.addItem(axis)

    def clearViewer(self):
        self.gl_view.clear()
        self.initScene()

    # Static Methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def formatNameSimple(landmark_id, waypoint_id=-1):
        if waypoint_id == -1:
            return "Landmark {}".format(landmark_id)
        else:
            return "Waypoint {}-{}".format(landmark_id, waypoint_id)

    @staticmethod
    def formatNameComplex(landmark, waypoint_id=-1, waypoint=None):
        if waypoint_id == -1:
            return "Landmark #{}: " \
                   "\n   \a Translation ({:.2}, {:.2}, {:.2})" \
                   "\n   \a Rotation    ({:.2}, {:.2}, {:.2})" \
                   "\n   \a Travel Time {:.2}".format(
                    landmark.id,
                    landmark.translation[0], landmark.translation[1], landmark.translation[2],
                    landmark.rotation[0], landmark.rotation[1], landmark.rotation[2],
                    landmark.travel_time)
        else:
            if waypoint.visible:
                return "       Waypoint #{}: R {:.2} | T ({:.2}, {:.2}, {:.2})".format(
                        waypoint_id, waypoint.rotation[0],
                        waypoint.translation[0], waypoint.translation[1], waypoint.translation[2])

    # Methods ----------------------------------------------------------------------------------------------------------
    def refresh(self):
        self.refreshListView()
        self.draw()

    def refreshListView(self):
        selected = self.selected_item
        self.fillListView()
        self.selected_item = selected

    def fillListView(self):
        self.model_landmarks.clear()
        self.selected_item = [-1, -1]

        for landmark in self.main_window.WaypointsPlanner.landmarks:
            self.model_landmarks.appendRow([ListItem(self.formatNameComplex(landmark))])
            for idx, waypoint in enumerate(landmark.waypoints):
                if waypoint.visible:
                    self.model_landmarks.appendRow([ListItem(self.formatNameComplex(landmark, idx, waypoint))])
        self.draw()

    def getSelectedIndexes(self, row):
        current_row = 0
        for landmark_idx, landmark in enumerate(self.main_window.WaypointsPlanner.landmarks):
            if (row >= current_row) and (row <= (current_row + len(landmark.waypoints))):
                return [landmark_idx, row - current_row - 1]

            current_row += 1
            current_row += len(landmark.waypoints)

        return [-1, -1]

    def selectItem(self, landmark_id, waypoint_id):
        if landmark_id == -1:
            return

        landmark = self.main_window.WaypointsPlanner.landmarks[landmark_id]

        self.qtRollSlider.setEnabled(True)
        self.qtPitchSlider.setEnabled(True)
        self.qtYawSlider.setEnabled(True)

        if waypoint_id == -1:
            self.original_position = [landmark.translation[0],
                                      landmark.translation[1],
                                      landmark.translation[2]]

            self.qtRollSlider.setValue(int(landmark.rotation[0]))
            self.qtPitchSlider.setValue(int(landmark.rotation[1]))
            self.qtYawSlider.setValue(int(landmark.rotation[2]))

            self.qtXLineEdit.setText(str(landmark.translation[0]))
            self.qtYLineEdit.setText(str(landmark.translation[1]))
            self.qtZLineEdit.setText(str(landmark.translation[2]))
        else:
            waypoint = landmark.waypoints[waypoint_id]
            self.original_position = [waypoint.translation[0],
                                      waypoint.translation[1],
                                      waypoint.translation[2]]

            self.qtRollSlider.setEnabled(False)
            self.qtRollSlider.setValue(0)
            self.qtPitchSlider.setEnabled(False)
            self.qtPitchSlider.setValue(0)
            self.qtYawSlider.setValue(int(waypoint.rotation[0]))

            self.qtXLineEdit.setText(str(waypoint.translation[0]))
            self.qtYLineEdit.setText(str(waypoint.translation[1]))
            self.qtZLineEdit.setText(str(waypoint.translation[2]))

        self.qtLandmarkIDLineEdit.setText(str(landmark.id))
        self.qtTravelTimeLineEdit.setText(str(landmark.travel_time))

        self.selected_item = [landmark_id, waypoint_id]
        self.draw()

    def translate(self, new_value, i):
        if self.selected_item == [-1, -1] or len(self.main_window.WaypointsPlanner.landmarks) == 0:
            return

        landmark_id = self.selected_item[0]
        waypoint_id = self.selected_item[1]

        landmark = self.main_window.WaypointsPlanner.landmarks[landmark_id]

        if waypoint_id == -1:
            self.main_window.WaypointsPlanner.landmarks[landmark_id].translation[i] = new_value
        else:
            waypoint = landmark.waypoints[waypoint_id]
            waypoint.translation[i] = new_value

        self.refreshListView()

    def rotate(self, new_value, i):
        if self.selected_item == [-1, -1] or len(self.main_window.WaypointsPlanner.landmarks) == 0:
            return

        landmark_id = self.selected_item[0]
        waypoint_id = self.selected_item[1]

        landmark = self.main_window.WaypointsPlanner.landmarks[landmark_id]

        if waypoint_id == -1:
            self.main_window.WaypointsPlanner.landmarks[landmark_id].rotation[i] = new_value
        else:
            if i != 2:  # only one rotation angle available -> Yaw
                return
            i = 0  # Yaw index is actually 0 for a waypoint

            waypoint = landmark.waypoints[waypoint_id]
            waypoint.rotation[0] = new_value

        self.refreshListView()

    def addNewLandmark(self):
        new_id = 0
        if len(self.main_window.WaypointsPlanner.landmarks) != 0:
            new_id = max([landmark.id for landmark in self.main_window.WaypointsPlanner.landmarks]) + 1
        waypoints = []
        landmark = Landmark(new_id,
                            np.array([self.qtXLineEdit.text(),
                                      self.qtYLineEdit.text(),
                                      self.qtZLineEdit.text()], dtype=float),
                            np.array([self.qtRollLineEdit.text(),
                                      self.qtPitchLineEdit.text(),
                                      self.qtYawLineEdit.text()],
                                     dtype=float),
                            waypoints,
                            self.qtTravelTimeLineEdit.text())
        self.main_window.WaypointsPlanner.landmarks.append(landmark)
        self.selectItem(new_id, -1)
        self.refreshListView()

    def addNewWaypoint(self):
        selected_item = self.selected_item[0]
        if selected_item == -1:
            return

        if selected_item != -1:
            landmark = self.main_window.WaypointsPlanner.landmarks[selected_item]
            new_waypoint = Waypoint(
                np.array([self.qtXLineEdit.text(),
                          self.qtYLineEdit.text(),
                          self.qtZLineEdit.text()], dtype=float),
                np.array([self.qtRollLineEdit.text(),
                          self.qtPitchLineEdit.text(),
                          self.qtYawLineEdit.text()], dtype=float))
            waypoint_id = len(landmark.waypoints)
            self.main_window.WaypointsPlanner.landmarks[selected_item].waypoints.append(new_waypoint)
            self.selectItem(selected_item, waypoint_id)
            self.refreshListView()

    # Draw Methods -----------------------------------------------------------------------------------------------------
    def drawLandmarks(self):
        routes = []
        for landmark_idx, landmark in enumerate(self.main_window.WaypointsPlanner.landmarks):
            landmark_axes = GLAxisItemWaypointsPlanner(obj_name=self.formatNameSimple(landmark.id),
                                                       landmark_id=landmark_idx,
                                                       waypoint_id=-1,
                                                       position=QVector3D(landmark.translation[0],
                                                                          landmark.translation[1],
                                                                          landmark.translation[2]),
                                                       orientation=QVector3D(landmark.rotation[0],
                                                                             landmark.rotation[1],
                                                                             landmark.rotation[2]))
            self.gl_view.addItem(landmark_axes)
            route = [landmark]
            route = self.drawWaypoints(landmark, route)
            routes.append(route)
        return routes

    def drawWaypoints(self, landmark, route):
        for waypoint_idx, waypoint in enumerate(landmark.waypoints):
            route.append(waypoint)
            if waypoint.visible:
                self.drawBox(name=self.formatNameSimple(landmark.id, waypoint_idx),
                             landmark_id=landmark.id,
                             waypoint_id=waypoint_idx,
                             point=waypoint)
        return route

    def drawLines(self, routes):
        for route in routes:
            for point_idx, [start_point, stop_point] in enumerate(zip(route[:-1], route[1:])):
                line = gl.GLLinePlotItem(pos=np.array([start_point.translation, stop_point.translation]),
                                         color=(0.0, 1.0, 0.0, 1.0),
                                         width=LINE_WIDTH)
                self.gl_view.addItem(line)

    def drawBox(self, name, landmark_id, waypoint_id, point):
        selected = (self.selected_item[1] == -1 and self.selected_item[0] == landmark_id) or \
                   (self.selected_item[1] == waypoint_id and self.selected_item[0] == landmark_id)
        color = (0., 50., 200.) if not selected else (200., 50., 0.)
        node = GLCubeItemWaypointsPlanner(
            box_name=name,
            landmark_id=landmark_id,
            waypoint_id=waypoint_id,
            position=QVector3D(point.translation[0] - NODE_WIDTH / 2.,
                               point.translation[1] - NODE_WIDTH / 2.,
                               point.translation[2] - NODE_WIDTH / 2.),
            color=color,
            size=QVector3D(NODE_WIDTH, NODE_WIDTH, NODE_WIDTH))
        self.gl_view.addItem(node)

    def drawMap(self):
        if not self.map.m_MapPoints:
            return

        # Parse map points
        points = []
        for map_point in self.map.m_MapPoints:
            points.append(map_point.m_Voxel.pt3d)

        points = np.array(points)
        points = points[:, :-1]
        scatter = gl.GLScatterPlotItem(pos=points, size=0.05, pxMode=False)
        self.gl_view.addItem(scatter)

        # Draw camera trajectory
        prev_kf_pos = None
        for key_frame in self.map.m_pMapKeyFrames:
            pos = key_frame.m_Absolute_Body_W.m_Transform[:3, 3]
            if prev_kf_pos is not None:
                line = gl.GLLinePlotItem(pos=np.array([prev_kf_pos, pos]),
                                         color=(1.0, 0.0, 0.0, 1.0),
                                         width=2)
                self.gl_view.addItem(line)
                self.map_objects = line
            prev_kf_pos = pos

    def draw(self):
        self.clearViewer()
        waypoints_route = self.drawLandmarks()
        self.drawLines(waypoints_route)
        self.drawMap()

    # Load Methods -----------------------------------------------------------------------------------------------------
    def loadWaypointsFile(self):
        self.main_window.WaypointsPlanner.clearLandmarks()
        self.main_window.WaypointsPlanner.loadLandmarks(self.main_window.waypoints_filename)
        self.fillListView()
        self.draw()

    def loadMapFile(self):
        map_storage = CMapStorage()
        map_storage.clear(self.map)
        map_storage.load(self.main_window.map_filename, self.map)
        self.draw()

    # Slots Buttons ----------------------------------------------------------------------------------------------------
    def on_saveToCSVButton_clicked(self):
        csv_file = QFileDialog.getSaveFileName(self, "Select CSV File",
                                               self.main_window.waypoints_filename, "CSV files (*.csv)")
        if csv_file is not None and csv_file[0] != "":
            self.main_window.waypoints_filename = csv_file[0]
            self.main_window.WaypointsPlanner.saveWaypoints(csv_file[0], self.qtQuaternionCheckBox.isChecked())

    def on_addLandmarkButton_clicked(self):
        self.addNewLandmark()

    def on_addWaypointButton_clicked(self):
        self.addNewWaypoint()

    def on_deleteWaypointButton_clicked(self):
        if self.selected_item != [-1, -1]:
            item = self.landmarks_selection_model.selection().indexes()[0]
            base_row = item.row()

            if self.selected_item[1] >= 0:
                self.main_window.WaypointsPlanner.landmarks[self.selected_item[0]].waypoints[
                    self.selected_item[1]].visible = False
                self.model_landmarks.takeRow(base_row)

                print("waypoint {} from Landmark {} was deleted".format(self.selected_item[1], self.selected_item[0]))
            else:
                # delete waypoints
                for _ in self.main_window.WaypointsPlanner.landmarks[self.selected_item[0]].waypoints:
                    self.model_landmarks.takeRow(base_row)

                del self.main_window.WaypointsPlanner.landmarks[self.selected_item[0]]

                # delete the landmark
                self.model_landmarks.takeRow(base_row)

                print("Landmark {} was deleted".format(self.selected_item[0]))

            self.refreshListView()

    def on_MoveUpButton_clicked(self):
        # TODO: Refactor
        if self.selected_item == [-1, -1]:
            return

        if self.selected_item[1] == -1:  # landmark
            if self.selected_item[0] == 0:
                return  # already at the top

            index = self.landmarks_selection_model.selection().indexes()[0]
            item = self.model_landmarks.itemFromIndex(index)
            base_row = item.row()

            landmarks = self.main_window.WaypointsPlanner.landmarks

            landmark_above = landmarks[self.selected_item[0] - 1]
            selectable_row = base_row - len(landmark_above.waypoints) - 1

            # swap
            landmarks[self.selected_item[0] - 1] = landmarks[self.selected_item[0]]
            landmarks[self.selected_item[0]] = landmark_above

            self.refreshListView()
            self.landmarks_selection_model.select(
                self.model_landmarks.indexFromItem(self.model_landmarks.item(selectable_row)),
                QItemSelectionModel.ClearAndSelect)
        else:
            if self.selected_item[1] == 0:
                return  # already at the top

            index = self.landmarks_selection_model.selection().indexes()[0]
            item = self.model_landmarks.itemFromIndex(index)
            base_row = item.row()

            waypoints = self.main_window.WaypointsPlanner.landmarks[self.selected_item[0]].waypoints

            tmp = waypoints[self.selected_item[1] - 1]
            waypoints[self.selected_item[1] - 1] = waypoints[self.selected_item[1]]
            waypoints[self.selected_item[1]] = tmp

            self.refreshListView()
            self.landmarks_selection_model.select(
                self.model_landmarks.indexFromItem(self.model_landmarks.item(base_row - 1)),
                QItemSelectionModel.ClearAndSelect)

    def on_MoveDownButton_clicked(self):
        # TODO: Refactor
        if self.selected_item == [-1, -1]:
            return

        if self.selected_item[1] == -1:  # landmark
            if self.selected_item[0] == len(self.main_window.WaypointsPlanner.landmarks) - 1:
                return  # already at the bottom

            index = self.landmarks_selection_model.selection().indexes()[0]
            item = self.model_map.itemFromIndex(index)
            base_row = item.row()

            landmarks = self.main_window.WaypointsPlanner.landmarks

            landmark_under = landmarks[self.selected_item[0] + 1]
            selectable_row = base_row + len(landmark_under.waypoints) + 1

            # swap
            landmarks[self.selected_item[0] + 1] = landmarks[self.selected_item[0]]
            landmarks[self.selected_item[0]] = landmark_under

            self.refreshListView()
            self.landmarks_selection_model.select(
                self.model_landmarks.indexFromItem(self.model_landmarks.item(selectable_row)),
                QItemSelectionModel.ClearAndSelect)
        else:
            if self.selected_item[1] == len(self.main_window.WaypointsPlanner.landmarks[self.selected_item[0]].waypoints) - 1:
                return  # already at the bottom

            index = self.landmarks_selection_model.selection().indexes()[0]
            item = self.model_landmarks.itemFromIndex(index)
            base_row = item.row()

            waypoints = self.main_window.WaypointsPlanner.landmarks[self.selected_item[0]].waypoints

            tmp = waypoints[self.selected_item[1] + 1]
            waypoints[self.selected_item[1] + 1] = waypoints[self.selected_item[1]]
            waypoints[self.selected_item[1]] = tmp

            self.refreshListView()
            self.landmarks_selection_model.select(
                self.model_landmarks.indexFromItem(self.model_landmarks.item(base_row + 1)),
                QItemSelectionModel.ClearAndSelect)

    # Slots GL Viewer --------------------------------------------------------------------------------------------------
    def on_gl_view_mouse_press(self, mouse_x, mouse_y, object_x, object_y, object_z, usable):
        selected = self.selected_item
        self.selected_item = [-1, -1]
        self.mouse_press_on_axis = -1
        self.mouse_is_down = True

        if usable:
            self.qtPoseLabel.setText("({:.2f}, {:.2f}) -> ({:.2f}, {:.2f}, {:.2f})"
                                     .format(mouse_x, mouse_y, object_x, object_y, object_z))

            # TODO Coords
            # self.qtXLineEdit.setText(str(object_x))
            # self.qtYLineEdit.setText(str(object_y))
            # self.qtZLineEdit.setText(str(object_z))

            # TODO Check
            if self.qtTopViewCheckBox.checkState() == Qt.Checked:
                self.selected_item = selected
                self.addNewWaypoint()
        else:
            self.qtPoseLabel.setText("({}, {})".format(mouse_x, mouse_y))

            if self.gl_view.last_selected_item is not None:
                current_row = 0
                for i in range(self.gl_view.last_selected_item.landmark_id):
                    current_row += 1
                    current_row += len(self.main_window.WaypointsPlanner.landmarks[i].waypoints)
                if self.gl_view.last_selected_item.waypoint_id != -1:
                    current_row += 1
                    current_row += self.gl_view.last_selected_item.waypoint_id

                self.landmarks_selection_model.select(
                    self.model_landmarks.indexFromItem(self.model_landmarks.item(current_row)),
                    QItemSelectionModel.ClearAndSelect)
                self.selectItem(self.gl_view.last_selected_item.landmark_id, self.gl_view.last_selected_item.waypoint_id)

            self.draw()

    def on_gl_view_mouse_release(self, mouse_x, mouse_y, object_x, object_y, object_z, usable):
        if usable:
            self.qtPoseLabel.setText("({:.2f}, {:.2f}) -> ({:.2f}, {:.2f}, {:.2f})"
                                     .format(mouse_x, mouse_y, object_x, object_y, object_z))

        self.mouse_press_on_axis = -1
        self.mouse_is_down = False

        self.draw()

    def on_gl_view_mouse_move(self, mouse_x, mouse_y, object_x, object_y, object_z, usable):
        if usable:
            self.qtPoseLabel.setText("({:.2f}, {:.2f}) -> ({:.2f}, {:.2f}, {:.2f})"
                                     .format(mouse_x, mouse_y, object_x, object_y, object_z))
            # TODO: Fix mouse move event
            # self.qtXLineEdit.setText(str(object_x))
            # self.qtYLineEdit.setText(str(object_y))
            # self.qtZLineEdit.setText(str(object_z))

    # Slots Check Boxes ------------------------------------------------------------------------------------------------
    def on_topViewCheckBox_stateChanged(self, state):
        if state == Qt.Checked:
            self.qtFrontViewCheckBox.setChecked(False)
            self.qtSideViewCheckBox.setChecked(False)
            self.gl_view.enable_move_camera_view = False
            self.prev_camera_position = self.gl_view.opts.copy()
            self.gl_view.setCameraPosition(distance=DISTANCE, elevation=90, azimuth=270)
        else:
            self.gl_view.setCameraPosition(distance=self.prev_camera_position['distance'],
                                           elevation=self.prev_camera_position['elevation'],
                                           azimuth=self.prev_camera_position['azimuth'])
            self.gl_view.enable_move_camera_view = True

    def on_frontViewCheckBox_stateChanged(self, state):
        if state == Qt.Checked:
            self.qtTopViewCheckBox.setChecked(False)
            self.qtSideViewCheckBox.setChecked(False)
            self.gl_view.enable_move_camera_view = False
            self.prev_camera_position = self.gl_view.opts.copy()
            self.gl_view.setCameraPosition(distance=DISTANCE, elevation=0, azimuth=180)
        else:
            self.gl_view.setCameraPosition(distance=self.prev_camera_position['distance'],
                                           elevation=self.prev_camera_position['elevation'],
                                           azimuth=self.prev_camera_position['azimuth'])
            self.gl_view.enable_move_camera_view = True

    def on_sideViewCheckBox_stateChanged(self, state):
        if state == Qt.Checked:
            self.qtTopViewCheckBox.setChecked(False)
            self.qtFrontViewCheckBox.setChecked(False)
            self.gl_view.enable_move_camera_view = False
            self.prev_camera_position = self.gl_view.opts.copy()
            self.gl_view.setCameraPosition(distance=DISTANCE, elevation=45, azimuth=45)
        else:
            self.gl_view.setCameraPosition(distance=self.prev_camera_position['distance'],
                                           elevation=self.prev_camera_position['elevation'],
                                           azimuth=self.prev_camera_position['azimuth'])
            self.gl_view.enable_move_camera_view = True

    # Slots Double Spin Boxes ------------------------------------------------------------------------------------------
    def on_coordinatesMinLineEdit_textChanged(self):
        self.qtXSlider.setMinimum(int(float(self.qtCoordsMinLineEdit.text()) / float(self.qtCoordsStepLineEdit.text())))
        self.qtYSlider.setMinimum(int(float(self.qtCoordsMinLineEdit.text()) / float(self.qtCoordsStepLineEdit.text())))
        self.qtZSlider.setMinimum(int(float(self.qtCoordsMinLineEdit.text()) / float(self.qtCoordsStepLineEdit.text())))

    def on_coordinatesMaxLineEdit_textChanged(self):
        self.qtXSlider.setMaximum(int(float(self.qtCoordsMaxLineEdit.text()) / float(self.qtCoordsStepLineEdit.text())))
        self.qtYSlider.setMaximum(int(float(self.qtCoordsMaxLineEdit.text()) / float(self.qtCoordsStepLineEdit.text())))
        self.qtZSlider.setMaximum(int(float(self.qtCoordsMaxLineEdit.text()) / float(self.qtCoordsStepLineEdit.text())))

    def on_coordinatesStepLineEdit_textChanged(self):
        self.on_coordinatesMinLineEdit_textChanged()
        self.on_coordinatesMaxLineEdit_textChanged()

    def on_angleStepLineEdit_textChanged(self):
        self.qtRollSlider.setMinimum(int(-180. / float(self.qtAngleStepLineEdit.text())))
        self.qtPitchSlider.setMinimum(int(-180. / float(self.qtAngleStepLineEdit.text())))
        self.qtYawSlider.setMinimum(int(-180. / float(self.qtAngleStepLineEdit.text())))

        self.qtRollSlider.setMaximum(int(180. / float(self.qtAngleStepLineEdit.text())))
        self.qtPitchSlider.setMaximum(int(180. / float(self.qtAngleStepLineEdit.text())))
        self.qtYawSlider.setMaximum(int(180. / float(self.qtAngleStepLineEdit.text())))

        self.on_rollSlider_valueChanged()
        self.on_pitchSlider_valueChanged()
        self.on_yawSlider_valueChanged()

    def on_xLineEdit_textChanged(self):
        if not len(self.original_position):
            return

        self.qtXSlider.setValue(
            int((float(self.qtXLineEdit.text()) - self.original_position[0]) / float(self.qtCoordsStepLineEdit.text())))

    def on_yLineEdit_textChanged(self):
        if not len(self.original_position):
            return

        self.qtYSlider.setValue(
            int((float(self.qtYLineEdit.text()) - self.original_position[1]) / float(self.qtCoordsStepLineEdit.text())))

    def on_zLineEdit_textChanged(self):
        if not len(self.original_position):
            return

        self.qtZSlider.setValue(
            int((float(self.qtZLineEdit.text()) - self.original_position[2]) / float(self.qtCoordsStepLineEdit.text())))

    def on_rollLineEdit_textChanged(self):
        self.qtRollSlider.setValue(float(self.qtRollLineEdit.text()) / float(self.qtAngleStepLineEdit.text()))

    def on_pitchLineEdit_textChanged(self):
        self.qtPitchSlider.setValue(float(self.qtPitchLineEdit.text()) / float(self.qtAngleStepLineEdit.text()))

    def on_yawLineEdit_textChanged(self):
        self.qtYawSlider.setValue(float(self.qtYawLineEdit.text()) / float(self.qtAngleStepLineEdit.text()))

    # Slots Sliders ----------------------------------------------------------------------------------------------------
    def on_xSlider_valueChanged(self, value):
        if not len(self.original_position):
            return

        new_value = round(self.original_position[0] + value * float(self.qtCoordsStepLineEdit.text()), 2)
        self.qtXLineEdit.setText(str(new_value))
        self.translate(new_value, 0)

    def on_ySlider_valueChanged(self, value):
        if not len(self.original_position):
            return

        new_value = round(self.original_position[1] + value * float(self.qtCoordsStepLineEdit.text()), 2)
        self.qtYLineEdit.setText(str(new_value))
        self.translate(new_value, 1)

    def on_zSlider_valueChanged(self, value):
        if not len(self.original_position):
            return

        new_value = round(self.original_position[2] + value * float(self.qtCoordsStepLineEdit.text()), 2)
        self.qtZLineEdit.setText(str(new_value))
        self.translate(new_value, 2)

    def on_rollSlider_valueChanged(self):
        self.qtRollLineEdit.setText(str(self.qtRollSlider.value() * float(self.qtAngleStepLineEdit.text())))
        self.rotate(self.qtRollSlider.value() * float(self.qtAngleStepLineEdit.text()), 0)

    def on_pitchSlider_valueChanged(self):
        self.qtPitchLineEdit.setText(str(self.qtPitchSlider.value() * float(self.qtAngleStepLineEdit.text())))
        self.rotate(self.qtPitchSlider.value() * float(self.qtAngleStepLineEdit.text()), 1)

    def on_yawSlider_valueChanged(self):
        self.qtYawLineEdit.setText(str(self.qtYawSlider.value() * float(self.qtAngleStepLineEdit.text())))
        self.rotate(self.qtYawSlider.value() * float(self.qtAngleStepLineEdit.text()), 2)

    # Slots List View --------------------------------------------------------------------------------------------------
    def on_landmarkList_selectionChanged(self):
        self.selected_item = [-1, -1]
        if len(self.landmarks_selection_model.selection().indexes()) != 0:
            item = self.landmarks_selection_model.selection().indexes()[0]

            self.selected_item = self.getSelectedIndexes(item.row())
            self.original_position.clear()
            self.selectItem(self.selected_item[0], self.selected_item[1])
        self.draw()

    def on_landmarkList_clicked(self, item_index):
        if self.selected_item == [-1, -1]:
            self.selected_item = self.getSelectedIndexes(item_index.row())

    def on_landmarkList_customContextMenuRequested(self, position):
        global_pos = self.qtLandmarksListView.mapToGlobal(position)
        item = self.model_landmarks.itemFromIndex(self.qtLandmarksListView.indexAt(position))

        if item is None:
            # right clicked empty space
            menu = QMenu()
            action_add = menu.addAction("Add landmark")
            menu.addSeparator()
            action_clear = menu.addAction("Clear landmarks")

            selected_action = menu.exec(global_pos)
            if selected_action == action_add:
                self.on_addLandmarkButton_clicked()
            elif selected_action == action_clear:
                self.main_window.WaypointsPlanner.landmarks.clear()
                self.model_landmarks.clear()
        else:
            indexes = self.getSelectedIndexes(item.row())
            menu = QMenu()
            action_add_w = menu.addAction("Add waypoint")
            action_move_up = menu.addAction("Move up")
            action_move_down = menu.addAction("Move down")
            action_add_l = menu.addAction("Add landmark")
            menu.addSeparator()
            action_delete = menu.addAction("Delete {}".format("landmark" if indexes[1] == -1 else "waypoint"))
            action_clear = menu.addAction("Clear landmarks")

            selected_action = menu.exec(global_pos)
            if selected_action == action_add_w:
                self.on_addWaypointButton_clicked()
            elif selected_action == action_add_l:
                self.on_addLandmarkButton_clicked()
            elif selected_action == action_delete:
                self.on_deleteWaypointButton_clicked()
            elif selected_action == action_clear:
                self.main_window.WaypointsPlanner.landmarks.clear()
                self.model_landmarks.clear()
            elif selected_action == action_move_up:
                self.on_MoveUpButton_clicked()
            elif selected_action == action_move_down:
                self.on_MoveDownButton_clicked()
