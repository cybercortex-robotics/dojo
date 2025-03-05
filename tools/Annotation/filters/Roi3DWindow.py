"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from tools.Annotation.filters.FilterWindowInterface import FilterWindowInterface
from tools.Annotation.items.Roi3DItem import Roi3DItem

from toolkit.sensors.pinhole_camera_sensor_model import PinholeCameraSensorModel
from toolkit.env.object_classes import ObjectClasses
from toolkit.vision.roi3d_utils import Box

from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import pyqtgraph.opengl as gl

from IPython.external.qt_for_kernel import QtGui
from PIL import Image

import numpy as np
import os
import shutil

current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")


class Roi3DWindow(FilterWindowInterface):
    # Signals ----------------------------------------------------------------------------------------------------------
    signal_activate_draw = pyqtSignal()

    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id, input_sources=None):
        super().__init__(filter_id=filter_id, input_sources=input_sources)
        uic.loadUi(os.path.join(ui_directory, 'Roi3DWindow.ui'), self)
        self.setWindowTitle("Roi3D Viewer")

        # Widget - children
        self.find_children()

        # Widget - Buttons
        self.qtButtonDeleteROI = self.findChild(QPushButton, 'qtButtonDeleteROI')
        self.qtButtonDrawROI = self.findChild(QPushButton, 'qtButtonDrawROI')
        self.qtRefreshPushButton = self.findChild(QPushButton, 'qtRefreshPushButton')

        # Widget spin box camera pos
        self.qtPitchDoubleSpinBox = self.findChild(QDoubleSpinBox, 'qtPitchDoubleSpinBox')
        self.qtRollDoubleSpinBox = self.findChild(QDoubleSpinBox, 'qtRollDoubleSpinBox')
        self.qtXCamDoubleSpinBox = self.findChild(QDoubleSpinBox, 'qtXCamDoubleSpinBox')
        self.qtYCamDoubleSpinBox = self.findChild(QDoubleSpinBox, 'qtYCamDoubleSpinBox')
        self.qtYawDoubleSpinBox = self.findChild(QDoubleSpinBox, 'qtYawDoubleSpinBox')
        self.qtZCamDoubleSpinBox = self.findChild(QDoubleSpinBox, 'qtZCamDoubleSpinBox')

        # Widget
        self.qtTabs = self.findChild(QTabWidget, 'tabWidget')
        self.qt3DArea = self.findChild(QWidget, 'qt3DArea')

        # Members
        self.object_classes = ObjectClasses(self.object_classes_path)
        self.gl_view = gl.GLViewWidget()
        self.scene = QGraphicsScene()
        self.graphics_2D_viewer = None
        self.assigned_camera = None
        self.original_image = None
        self.draw_class = None
        self.selected_box = None
        self.image_timestamps_dict = {}
        self.classes = list()
        self.boxes_objects = []
        self.boxes_objects_3D = []
        self.boxes_gl = []
        self.pen_size = 5
        self.initial_data = []

        # Initialization
        self.init_connections()
        self.init_window()
        self.init_tree_widget()

        self.init_combo_overlay()
        self.init_3d_gl_viewer()
        self.init_camera_params()

        self.init_splitter()

    # Initialization ---------------------------------------------------------------------------------------------------
    def init_camera_params(self):
        if self.calibration_path is None:
            self.qtButtonDrawROI.setEnabled(False)
            self.qtButtonDeleteROI.setEnabled(False)
            self.qtButtonSave.setEnabled(False)

            return

        self.assigned_camera = PinholeCameraSensorModel(self.calibration_path)

        self.qtRollDoubleSpinBox.setValue(self.assigned_camera.rotation[0])
        self.qtPitchDoubleSpinBox.setValue(self.assigned_camera.rotation[1])
        self.qtYawDoubleSpinBox.setValue(self.assigned_camera.rotation[2])

        self.qtXCamDoubleSpinBox.setValue(self.assigned_camera.translation[0])
        self.qtYCamDoubleSpinBox.setValue(self.assigned_camera.translation[1])
        self.qtZCamDoubleSpinBox.setValue(self.assigned_camera.translation[2])

    def init_3d_gl_viewer(self):
        z_grid = gl.GLGridItem()
        z_grid.scale(10, 10, 10)
        self.gl_view.addItem(z_grid)

        axis = gl.GLAxisItem(size=QtGui.QVector3D(10, 10, 10), glOptions='opaque')
        self.gl_view.addItem(axis)
        axis = gl.GLAxisItem()
        self.gl_view.addItem(axis)
        self.gl_view.setCameraPosition(distance=300)

        layout = QGridLayout()
        self.qt3DArea.setLayout(layout)
        self.qt3DArea.layout().addWidget(self.gl_view)

    # Inherited Initialization Methods ---------------------------------------------------------------------------------
    def init_window(self):
        self.qtWarning.setText("ATTENTION: Class index out of bounds. Please check your classes file")
        self.qtWarning.setStyleSheet("QLabel {background-color:red}")
        self.show_warning_text(False)

        self.image_timestamps_dict = self._map_timestamps_to_dict(self.parser, self.filter_id_image)
        self.image_timestamps = self.parser.get_all_timestamps(filter_id=self.filter_id_image)

        self.qtGraphicsView.setScene(self.scene)
        self.update_save_button(False)

    def init_connections(self):
        self.qtComboOverlay.currentTextChanged.connect(self.on_change_text_combo)
        self.qtTabs.tabBarClicked.connect(self.on_tab_clicked)
        self.qtTabs.currentChanged.connect(self.on_main_splitter_moved)
        self.qtButtonDrawROI.clicked.connect(self.on_click_button_draw)
        self.qtButtonDeleteROI.clicked.connect(self.on_click_button_delete)
        self.qtButtonSave.clicked.connect(self.on_click_button_save)
        self.qtClasses.itemClicked.connect(self.on_select_class)
        self.scene.selectionChanged.connect(self.on_highlighted_class)
        self.qtRefreshPushButton.clicked.connect(self.on_click_button_refresh)

    def init_tree_widget(self):
        self.qtClasses.setHeaderHidden(True)
        self.qtClasses.setColumnCount(3)
        self.qtClasses.setColumnWidth(0, 10)
        self.qtClasses.setColumnWidth(1, 120)
        self.qtClasses.setColumnWidth(2, 30)
        self.qtClasses.clear()
        self.classes.clear()
        color = self.object_classes.colormap()

        # Init object classes and assign colors.
        idx = 0
        for obj_class in self.object_classes.object_classes:
            class_name = obj_class[1]
            self.classes.append(class_name)

            class_color_red = color[idx][2] / 255.
            class_color_green = color[idx][1] / 255.
            class_color_blue = color[idx][0] / 255.
            idx += 1

            qColor = QColor(0, 0, 0)
            qColor.setRedF(class_color_red)
            qColor.setGreenF(class_color_green)
            qColor.setBlueF(class_color_blue)

            parent = QtWidgets.QTreeWidgetItem(self.qtClasses)
            parent.setText(0, str(idx))
            parent.setText(1, str(class_name))
            parent.setBackground(2, QBrush(qColor))

    # Static Methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _map_timestamps_to_dict(parser, filter_id):
        """
        Maps all timestamps to dictionary for performance reasons.
        :param parser: Datastream file parser depending on the filter type.
        :param filter_id: Filter type.
        :return: Dictionary containing entries of shape -> (line_index, ts_start, ts_stop)
        """
        return dict([(item[2], item) for item in parser.get_all_timestamps(filter_id=filter_id)])

    @staticmethod
    def _int_to_rgb(color_int):
        """Converts a int value into r,g,b color values in range [0, 1]"""
        rgb = (color_int // 256 // 256 % 256, color_int // 256 % 256, color_int % 256, 256)
        rgb_normalized = tuple(map(lambda x: x / 256, rgb))
        return rgb_normalized

    # Inherited Methods ------------------------------------------------------------------------------------------------
    def clear(self):
        """Removes items from viewer on update"""
        for gl_box in self.boxes_gl:
            if gl_box in self.gl_view.items:
                self.gl_view.removeItem(gl_box)

        # Clear boxes objects list
        if self.boxes_objects:
            self.boxes_objects.clear()
        if self.boxes_objects_3D:
            self.boxes_objects_3D.clear()

        # Clear image scene
        self.scene.clear()

    def get_class_item(self, class_idx):
        if class_idx >= len(self.object_classes.object_classes):
            return None

        class_name = self.object_classes.object_classes[class_idx][1]
        items = self.qtClasses.findItems(class_name, Qt.MatchContains, 1)

        for item in items:
            if item.text(1) == class_name:
                return item
        self.signal_error_signal.emit("Invalid object class file!", self.filter_id)
        return None

    # Private Methods ROIs ---------------------------------------------------------------------------------------------
    def _load_rois(self, rois):
        """
        Load all roi data related to a frame based on it's timestamp.
        :param rois: List<List<str>>: Each roi parameter is being read as a string.
        """
        self.show_number_of_items(str(len(rois)))
        for roi in rois:
            frame_id, roi_id, box_center, box_angles, box_whl, obj_class = self._parse_roi_from_text(roi)
            self.frame_id = frame_id
            self.qtFrameID.setText(str(self.frame_id))
            self.last_item_id = int(roi_id)

            self._create_box(frame_id=frame_id,
                             roi_id=roi_id,
                             box_center=box_center,
                             box_whl=box_whl,
                             box_orientation=box_angles,
                             box_class_index=obj_class)

    def _draw_3d_rois(self, boxes: list):
        """
        Given the bounding box parameters, instatiates and displays box in 3D viewer.
        :param boxes: Box object to be rendered.
        """
        for bbox in boxes:
            gl_box = bbox.get_wireframe_box(1)
            self.boxes_gl.append(gl_box)
            self.gl_view.addItem(gl_box)

    def _create_box(self, frame_id, roi_id, box_center, box_whl, box_orientation, box_class_index):
        if self.assigned_camera is None:
            self.signal_error_signal.emit("Camera is none! Filter image id {}!"
                                          .format(self.filter_id_image), self.filter_id)
            return

        if self.original_image is None:
            self.signal_error_signal.emit("Image is None!", self.filter_id)
            return

        color3d = self._find_class_color(box_class_index)
        color2d = self._get_class_color(box_class_index)
        box = Box(frame_index=frame_id, roi_index=roi_id, center=box_center, size=box_whl, orientation=box_orientation,
                  color=color3d)

        box_copy = Roi3DItem.copy(box)
        self.boxes_objects_3D.append(box_copy)

        pen_box = QPen(color2d, self.pen_size)
        bbox = Roi3DItem(box=box,
                         pen=pen_box,
                         cls=box_class_index,
                         assigned_camera=self.assigned_camera,
                         image_size=self.original_image.shape[:2])
        self.boxes_objects.append(bbox)

        bbox.set_connection(self.update_save_button, self.on_move_box)
        bbox.draw_box(bbox.get_corners())
        self.selected_box = bbox

        pen = QPen(Qt.red)
        self.scene.addEllipse(box_center[0], box_center[2], 10, 10, pen)

        for line in bbox.lines:
            self.scene.addItem(line)
        self.on_main_splitter_moved()

    def _parse_roi_from_text(self, roi_string):
        """
        Parses the ROI string as it is read from descriptor.csv file.
        :param roi_string: str: One record of [frame_index, roi_index, x, y, z, w, h, l, roll, pitch, yaw] as strings.
        """
        try:
            frame_id = int(float(roi_string[0]))
            roi_id = int(float(roi_string[1]))
            obj_class = int(roi_string[2])
            box_center = [float(i) for i in roi_string[3:6]]
            box_whl = [float(i) for i in roi_string[6:9]]
            box_angles = [float(i) for i in roi_string[9:]]
            return frame_id, roi_id, box_center, box_angles, box_whl, obj_class
        except IndexError:
            self.signal_error_signal.emit("Malformed record in framebased descriptor.", self.filter_id)
            print("Malformed record in framebased descriptor.")
            exit(-1)

    # Private Methods Classes ------------------------------------------------------------------------------------------
    def _get_class_color(self, class_idx):
        item = self.get_class_item(class_idx)
        if item is not None:
            return item.background(2)
        else:
            self.signal_error_signal.emit("Invalid object class file!", self.filter_id)
            self.show_warning_text(True)
            return QBrush(Qt.white)

    def _find_class_color(self, class_id):
        """Selects color assigned to class name from the GUI"""
        if class_id >= len(self.object_classes.object_classes):
            self.show_warning_text(True)
            self.signal_error_signal.emit("Invalid object class file!", self.filter_id)
            return 1, 1, 1, 1  # white
        class_name = self.object_classes.object_classes[class_id][1]
        items = self.qtClasses.findItems(class_name, Qt.MatchContains, 1)

        if len(items):
            if str(items[0].text(1)) == str(class_name):
                return self._int_to_rgb(items[0].background(2).color().rgb())
        self.show_warning_text(True)
        self.signal_error_signal.emit("Invalid object class file!", self.filter_id)
        return 1, 1, 1, 1  # white

    # Private Methods --------------------------------------------------------------------------------------------------
    def _load_current_image(self, img_format='cv2'):
        """
        Load current image based on the current timestamp.
        """
        image_timestamp = self.image_timestamps_dict.get(self.ts_sync)
        _, _, _, image_array, _ = self.parser.parse_line(filter_id=self.filter_id_image,
                                                         line_index=image_timestamp[0])
        if img_format == 'cv2':
            return image_array
        else:
            return Image.fromarray(np.asarray(image_array))

    def _update_renderer(self):
        """Once new data is loaded, the renderer needs to be updated."""
        self.on_tab_clicked(0)
        self.on_tab_clicked(1)

    # Slots ------------------------------------------------------------------------------------------------------------
    def on_click_button_refresh(self):
        if self.assigned_camera is None:
            return

        CyC_rot_tr_str = "{},{},{},{},{},{}".format(self.qtRollDoubleSpinBox.value(),
                                                      self.qtPitchDoubleSpinBox.value(),
                                                      self.qtYawDoubleSpinBox.value(),
                                                      self.qtXCamDoubleSpinBox.value(),
                                                      self.qtYCamDoubleSpinBox.value(),
                                                      self.qtZCamDoubleSpinBox.value())
        self.assigned_camera.set_camera_pose(CyC_rot_tr_str)
        self.assigned_camera.generate_matrix()

        self.clear()
        self.on_load_data(self.initial_data[0], self.initial_data[1])

    def on_box_changed(self):
        for gl_box in self.boxes_gl:
            if gl_box in self.gl_view.items:
                self.gl_view.removeItem(gl_box)
        self.boxes_objects_3D.clear()
        for box in self.boxes_objects:
            det = box.get_item_details()
            details = det.split(',')
            frame_id, roi_id, box_center, box_angles, box_whl, obj_class = self._parse_roi_from_text(details)
            color3d = self._find_class_color(obj_class)
            box_copy = Box(frame_index=frame_id, roi_index=roi_id, center=box_center, size=box_whl,
                           orientation=box_angles, color=color3d)
            self.boxes_objects_3D.append(box_copy)
        self._draw_3d_rois(self.boxes_objects_3D)

    def on_tab_clicked(self, index):
        if index == 0:
            self.on_box_changed()
        elif index == 1:
            pass

    def on_move_box(self, _type):
        if len(self.scene.selectedItems()):
            line = self.scene.selectedItems()[0]
            for bbox in self.boxes_objects:
                if line in bbox.lines:
                    if _type == 1:
                        bbox.move_box(dx=0.2)
                    elif _type == 2:
                        bbox.move_box(dx=-0.2)
                    elif _type == 3:
                        bbox.move_box(dy=0.2)
                    elif _type == 4:
                        bbox.move_box(dy=-0.2)
                    elif _type == 5:
                        bbox.move_box(dz=0.2)
                    elif _type == 6:
                        bbox.move_box(dz=-0.2)
                    bbox.draw_box(bbox.get_corners())
                    self.update_save_button(True)
        self.on_box_changed()

    # Inherited Slots Classes ------------------------------------------------------------------------------------------
    def on_highlighted_class(self):
        try:
            self.qtClasses.clearSelection()
            self.show_item_id()
            if self.scene and len(self.scene.selectedItems()) > 0:
                clicked_line = self.scene.selectedItems()[0]
                for box in self.boxes_objects:
                    if box.box.roi_index == clicked_line.get_box_id():
                        self.show_item_id(box.box.roi_index)
                        class_id = box.cls
                        try:
                            item = self.get_class_item(class_id)
                            item.setSelected(True)
                        except IndexError:
                            self.signal_error_signal.emit("Class id {} not found in tree!"
                                                          .format(class_id), self.filter_id)
                        if self.selected_box is not None:
                            if clicked_line not in self.selected_box.lines:
                                self.selected_box.hide_axis()
                        self.selected_box = box
                        break
        except RuntimeError:
            self.signal_error_signal.emit("RuntimeError!", self.filter_id)

    def on_select_class(self, it):
        self.draw_class = it
        if len(self.scene.selectedItems()):
            line = self.scene.selectedItems()[0]
            for bbox in self.boxes_objects:
                if line in bbox.lines:
                    bbox.box.set_color(self._get_class_color(self.qtClasses.currentIndex().row()))
                    bbox.cls = self.qtClasses.currentIndex().row()
                    bbox.pen = QPen(bbox.box.color, self.pen_size)
                    self.update_save_button(True)

    # Inherited Slots Load Data ----------------------------------------------------------------------------------------
    def on_load_data(self, line_index, ts_sync_timestamp):
        """
        Called on each timestamp update. Reads data from sync descriptor.
        :param line_index: <int> Represents the line index at which to read.
        :param ts_sync_timestamp: <str> Timestamp by which the stream is synchronized.
        """
        self.show_warning_text(False)
        data, img = super().on_load_data(line_index, ts_sync_timestamp)
        self.show_runtime_values(frame_id=line_index - 1)
        self.clear()
        self.initial_data = [line_index, ts_sync_timestamp]

        if self.calibration_path is None:
            self.signal_error_signal.emit("Invalid calibration file for filter id {}!"
                                          .format(self.filter_id_image), self.filter_id)

            self.qtWarning.setText("ATTENTION: Calibration file for filter id {} not found!".format(self.filter_id_image))
            self.qtWarning.setStyleSheet("QLabel {background-color:red}")
            self.show_warning_text(True)

        if img is None or data is None:
            return

        ts_stop_rois, sampling_time_rois, rois = data
        ts_stop, sampling_time, ts_image, cv_image, _ = img
        self.original_image = cv_image
        self.load_image(cv_image)

        self.show_runtime_values(ts_stop=ts_stop_rois, ts_sampling_time=sampling_time_rois,
                                 ts_image=ts_image, frame_id=self.frame_id)

        if rois is None:
            self.signal_warning_signal.emit("Rois NONE for line index {}, timestamp sync {}!"
                                            .format(line_index, ts_sync_timestamp), self.filter_id)
            return

        if len(rois):
            self._load_rois(rois)
        else:
            self.signal_warning_signal.emit("Empty rois!", self.filter_id)

        # Update
        self._update_renderer()
        self.on_main_splitter_moved()

    # Inherited Slots Draw/Delete/Save ---------------------------------------------------------------------------------
    def on_click_button_draw(self):
        if self.qtTabs.currentIndex() != 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("Please select the 2D tab!")
            msg.exec()
        else:
            if self.draw_enabled:
                self.draw_enabled = False
                self.signal_activate_draw.emit()
                self.qtButtonDrawROI.setText("Draw Box")
            else:
                if self.draw_class is None:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Warning")
                    msg.setText("Select a class!")
                    msg.exec()
                else:
                    starting_position = [5.0, 0.0, 0.5]
                    frame_id = int(self.frame_id)
                    box_id = int(self.last_item_id)
                    box_class = self.qtClasses.currentIndex().row()
                    box_center = starting_position
                    box_whl = [1.0, 1.0, 1.0]
                    box_angles = [0.0, 0.0, 0.0]
                    self._create_box(frame_id=frame_id,
                                     roi_id=box_id,
                                     box_center=box_center,
                                     box_whl=box_whl,
                                     box_orientation=box_angles,
                                     box_class_index=box_class)
                    self.draw_enabled = True
                    self.signal_activate_draw.emit()
                    self.qtButtonDrawROI.setText("Stop Drawing Box")
                    self.update_save_button(True)
                    self.on_click_button_draw()

    def on_click_button_delete(self):
        if self.qtTabs.currentIndex() != 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("Please select the 2D tab!")
            msg.exec()
        else:
            if len(self.scene.selectedItems()):
                line = self.scene.selectedItems()[0]
                for bbox in self.boxes_objects:
                    if line in bbox.lines:
                        for line in bbox.lines:
                            self.scene.removeItem(line)
                        box_id = bbox.box.roi_index
                        bbox_3d = None
                        for box_3d in self.boxes_objects_3D:
                            if box_3d.roi_index == box_id:
                                bbox_3d = box_3d
                        self.boxes_objects_3D.remove(bbox_3d)
                        self.boxes_objects.remove(bbox)
                        self.update_save_button(True)
                        self.selected_box = None
        self.on_box_changed()
        self.on_main_splitter_moved()

    def on_click_button_save(self):
        base_path = self.parser.get_parser_base_path(self.filter_id)
        file_path = os.path.join(base_path, 'framebased_data_descriptor.csv')
        file_copy_path = os.path.join(base_path, 'framebased_data_descriptor_copy.csv')
        shutil.copyfile(file_path, file_copy_path)
        wrote = False

        with open(file_path, "r") as f_in:
            lines = f_in.readlines()

        with open(file_path, "w") as f_out:
            for line in lines:
                if line == "frame_id,roi_id,cls,x,y,z,w,h,l,roll,pitch,yaw\n":
                    f_out.write(line)
                else:
                    line_frame_id = int(line.split(",")[0])
                    if line_frame_id == self.frame_id:
                        if wrote is False:
                            wrote = True
                            self.boxes_objects = sorted(self.boxes_objects, key=lambda item: item.box.roi_index)
                            for box in self.boxes_objects:
                                f_out.write(box.get_item_details())
                    else:
                        f_out.write(line)
            if wrote is False:
                for box in self.boxes_objects:
                    f_out.write(box.get_item_details())
        os.remove(file_copy_path)
        self.on_main_splitter_moved()
        self.update_save_button(False)
        self._draw_3d_rois(self.boxes_objects_3D)

    # Inherited Slots Widget -------------------------------------------------------------------------------------------
    def on_main_splitter_moved(self):
        if self.original_image is not None:
            self.scene.setSceneRect(0, 0, self.original_image.shape[1], self.original_image.shape[0])
        self.qtGraphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
