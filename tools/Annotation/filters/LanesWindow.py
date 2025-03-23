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
from tools.Annotation.items.DojoGraphicScene import DojoGraphicScene
from tools.Annotation.items.LanesPointItem import LanesPointItem

from toolkit.vision.lane_utils import generate_lane, lane_3d_from_2d
import toolkit.sensors.pinhole_camera_sensor_model as pinhole

from toolkit.env.object_classes import ObjectClasses
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import numpy as np

import os
import cv2
import shutil

current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")

default_pen_size_ratio = 80
default_thickness_ratio = 160


class LanesWindow(FilterWindowInterface):
    # Signals ----------------------------------------------------------------------------------------------------------
    signal_save_finished = pyqtSignal(object, object)

    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id, input_sources=None):
        super().__init__(filter_id, input_sources)
        uic.loadUi(os.path.join(ui_directory, 'LanesWindow.ui'), self)
        self.setWindowTitle("Lanes Viewer")

        # Widget - children
        self.find_children()

        self.qtCamSegmented = self.findChild(QGraphicsView, 'qtCamSegmented')
        self.qtCamInstances = self.findChild(QGraphicsView, 'qtCamInstances')
        self.qtCamThetas = self.findChild(QGraphicsView, 'qtCamThetas')

        # Widget - Buttons
        self.qtButtonGenerateThetas = self.findChild(QPushButton, 'qtButtonGenerateThetas')
        self.qtButtonDrawLane = self.findChild(QPushButton, 'qtButtonDrawLane')
        self.qtButtonDeleteLane = self.findChild(QPushButton, 'qtButtonDeleteLane')
        self.qtButtonClearFrame = self.findChild(QPushButton, 'qtButtonClearFrame')
        self.qtButtonCopyFrame = self.findChild(QPushButton, 'qtButtonCopyFrame')
        self.qtButtonPasteFrame = self.findChild(QPushButton, 'qtButtonPasteFrame')

        # Widget
        self.qtSegSplitter = self.findChild(QSplitter, 'qtSegSplitter')
        self.qtTabs = self.findChild(QTabWidget, 'tabWidget')
        self.qtHBox = self.findChild(QCheckBox, 'qtHBox')

        # Members
        self.object_classes = ObjectClasses(object_classes_file=None)
        self.frame_based_header = "timestamp_stop,lane_id,points,theta_0,theta_1,theta_2,theta_3\n"
        self.scene = DojoGraphicScene()
        self.selected_item = None
        self.original_image = None
        self.temp_point = None
        self.laneObjects = {}
        self.thetas = {}
        self.height_index = 0
        self.pen_size = 1
        self.thickness = 1
        self.h_lines = []
        self.cam = None
        self.lines_height = 0
        self.lines_number = 0
        self.frame_clipboard = {}

        # Initialization
        self.init_connections()
        self.init_window()
        self.init_splitter()
        self.init_camera()

    # Initialization Methods -------------------------------------------------------------------------------------------
    def init_camera(self):
        if self.calibration_path == "" or self.calibration_path is None:
            self.signal_error_signal.emit("Invalid calibration file for image filter id {}!"
                                          .format(self.filter_id_image), self.filter_id)
            return

        self.cam = pinhole.PinholeCameraSensorModel(self.calibration_path)

        # Fixed theta generation setting rotation
        # roll 90. pitch 180. yaw 90.
        self.cam.rotation = [90., 180., 90.]
        self.cam.generate_matrix()

        self.lines_height = self.cam.lane_filter_lines_height
        self.lines_number = self.cam.lane_filter_lines_number

    # Inherited Initialization Methods ---------------------------------------------------------------------------------
    def init_window(self):
        self.image_timestamps = self.parser.get_all_timestamps(filter_id=self.filter_id_image)
        self.synced_timestamps = self.parser.get_records_synced(filter_ids=[self.filter_id_image, self.filter_id])
        self.init_combo_overlay()
        self.on_change_text_combo()

        self.qtGraphicsView.setScene(self.scene)
        self.qtGraphicsView.setMouseTracking(True)
        self.update_save_button(False)

        self.qtCamSegmented.setScene(QGraphicsScene())
        self.qtCamInstances.setScene(QGraphicsScene())
        self.qtCamThetas.setScene(QGraphicsScene())

        self.on_main_splitter_moved()

    def init_connections(self):
        self.scene.selectionChanged.connect(self.on_selection_changed)
        self.qtButtonSave.clicked.connect(self.on_click_button_save)
        self.qtButtonDrawLane.clicked.connect(self.on_click_button_draw)
        self.qtButtonDeleteLane.clicked.connect(self.on_click_button_delete)
        self.qtButtonGenerateThetas.clicked.connect(self.on_click_generate_thetas)

        self.qtButtonClearFrame.clicked.connect(self.on_click_button_clear_frame)
        self.qtButtonCopyFrame.clicked.connect(self.on_click_button_copy_frame)
        self.qtButtonPasteFrame.clicked.connect(self.on_click_button_paste_frame)

        self.qtHBox.stateChanged.connect(self.on_change_check_box)
        self.scene.signal_draw_new_point.connect(self.on_draw_new_point)
        self.scene.signal_cursor_position.connect(self.on_move_temp_point)
        self.scene.signal_right_click.connect(self.on_delete_point)
        self.splitter.splitterMoved.connect(self.on_main_splitter_moved)
        self.qtTabs.currentChanged.connect(self.on_main_splitter_moved)
        self.qtSegSplitter.splitterMoved.connect(self.on_main_splitter_moved)

        self.signal_activate_draw.connect(self.scene.on_active_draw)
        self.signal_save_finished.connect(self.on_save_finished)

    # Inherited Events -------------------------------------------------------------------------------------------------
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            if self.height_index < self.lines_number - 1:
                self.height_index += 1

                if self.temp_point is not None:
                    self.temp_point.setY(self._get_line_h(self.height_index))
        else:
            if self.height_index > 0:
                self.height_index -= 1

                if self.temp_point is not None:
                    self.temp_point.setY(self._get_line_h(self.height_index))

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.UpArrow or key == Qt.Key_W:
            if self.height_index < self.lines_number - 1:
                self.height_index += 1

                if self.temp_point is not None:
                    self.temp_point.setY(self._get_line_h(self.height_index))

        if key == Qt.DownArrow or key == Qt.Key_S:
            if self.height_index > 0:
                self.height_index -= 1

                if self.temp_point is not None:
                    self.temp_point.setY(self._get_line_h(self.height_index))

        super().keyPressEvent(event)

    # Inherited Methods ------------------------------------------------------------------------------------------------
    def clear(self):
        if self.draw_enabled:
            self.on_click_button_draw()

        if self.qtButtonSave.isEnabled():
            self.on_click_button_save()

        self.update_save_button(False)
        self.scene.selectedItems().clear()
        self.selected_item = None

        # Clear horizontal lines
        for line in self.h_lines:
            self.scene.removeItem(line)
        self.h_lines.clear()

        # Clear points
        for lane in self.laneObjects.values():
            for point in lane:
                if point is not None:
                    self.scene.removeItem(point)
        self.laneObjects.clear()

        self.scene.removeItem(self.temp_point)
        self.temp_point = None

        # Thetas
        self.thetas.clear()

        # Scenes
        self.scene.clear()
        self.qtCamSegmented.scene().clear()
        self.qtCamInstances.scene().clear()
        self.qtCamThetas.scene().clear()

    def get_item_details(self, lane_id):
        if lane_id not in self.laneObjects.keys():
            return ''

        lane = []
        for item in self.laneObjects[lane_id]:
            if item is None:
                lane.append(-1)
            else:
                lane.append(int(item.pos().x()))

        return "{},{},{}\n".format(self.qtTsStop.text(),
                                   int(lane_id),
                                   str(lane).replace(",", ""))

    # Private Methods - Horizontal Lines -------------------------------------------------------------------------------
    def _get_line_h(self, num):
        height = self.original_image.shape[0]
        return int(height - (num * self.lines_height * height / self.lines_number))

    def _create_lines(self):
        if self.original_image is None:
            return
        self.scene.setSceneRect(0, 0, self.original_image.shape[1], self.original_image.shape[0])

        for i in range(self.lines_number):
            h = self._get_line_h(i)
            h_line = QGraphicsLineItem(1, h + self.pen_size/2, self.original_image.shape[1] - 1, h + self.pen_size/2)
            line_pen = QPen(QColor(255, 0, 0, 100))
            line_pen.setWidth(self.pen_size // 4)
            h_line.setPen(line_pen)
            self.h_lines.append(h_line)
            self.scene.addItem(h_line)

    def _show_lines(self, value):
        for h in self.h_lines:
            h.setVisible(value)

    # Private Methods - Points -----------------------------------------------------------------------------------------
    def _create_temp_point(self):
        self.temp_point = LanesPointItem(-1, 0, 0, QPen(QColor(255, 255, 0, 150)), QBrush(QColor(255, 255, 0, 150)),
                                         0, 0, self.pen_size, self.pen_size)
        self.scene.addItem(self.temp_point)
        self.temp_point.hide()

    def _draw_point(self, x, y):
        color = self.object_classes.colormap()
        idx = self.last_item_id
        # pen = QPen(QColor(color[idx][0], color[idx][1], color[idx][2], 175))
        pen = QPen(QColor(0, 0, 0, 255))
        brush = QBrush(QColor(color[idx][0], color[idx][1], color[idx][2], 190))
        point = LanesPointItem(idx, x, y, pen, brush, 0, 0, self.pen_size, self.pen_size)
        return point

    def _delete_point(self):
        lane_id = self.last_item_id

        point = self.laneObjects[lane_id][self.height_index]
        if point is None:
            return
        self.scene.removeItem(point)
        self.laneObjects[lane_id][self.height_index] = None
        self.update_save_button(True)

    # Private Methods - Lanes ------------------------------------------------------------------------------------------
    def _load_lanes(self, points):
        p_lens = [len(points[key]) for key in points.keys()]

        # Check whether the points have the same size
        if len(np.unique(p_lens)) > 1:
            self.signal_error_signal.emit("Points are corrupted!", self.filter_id)
            return False

        # Check whether the len of points it is equal to the default_lines_number
        if p_lens != [] and p_lens[0] != self.lines_number:
            self.signal_error_signal.emit(
                "Configuration changed, check the default values!", self.filter_id)
            return False

        for lane_id, lane in points.items():
            self.laneObjects[lane_id] = []
            for i, point in enumerate(lane):
                if point == -1:
                    self.laneObjects[lane_id].append(None)
                elif point != -1:
                    self.last_item_id = lane_id
                    qt_point = self._draw_point(point, self._get_line_h(i))
                    self.laneObjects[lane_id].append(qt_point)
                    self.scene.addItem(qt_point)
        return True

    def _check_lane(self, lane_id):
        if (len(set(self.laneObjects[lane_id])) - 1) < 2:  # Have 2 points or more
            return False
        return True

    def _delete_lane(self, lane_id):
        self.update_save_button(True)

        for i, point in enumerate(self.laneObjects[lane_id]):
            if point is not None:
                self.scene.removeItem(point)
                self.laneObjects[lane_id][i] = None

        self.laneObjects.pop(lane_id)

        self.show_number_of_items(len(self.laneObjects.keys()))
        self.qtItemID.setText(str(-1))

    # Private Methods - Images -----------------------------------------------------------------------------------------
    def _generate_images(self):
        org_img = self.original_image[..., ::-1].copy()
        sem_seg_img = np.zeros((org_img.shape[0], org_img.shape[1], 3), dtype=np.uint8)
        inst_img = np.zeros((org_img.shape[0], org_img.shape[1], 3), dtype=np.uint8)

        for lane_id in self.laneObjects.keys():
            lane = []
            for item in self.laneObjects[lane_id]:
                if item is None:
                    lane.append(-1)
                else:
                    if int(item.pos().x()) != -1:
                        lane.append(int(item.pos().x()))
                    else:
                        lane.append(int(0))
            x = []
            y = []
            for i in range(len(lane)):  # for each point of the lane
                if lane[i] == -1:
                    continue
                x.append(lane[i])
                y.append(self._get_line_h(i))

            # Bring point coords to center of point
            for i in range(len(x)):
                x[i] += self.pen_size // 2
                y[i] += self.pen_size // 2

            for i in range(1, len(x)):
                cv2.line(sem_seg_img, (x[i - 1], y[i - 1]), (x[i], y[i]), (1, 1, 1), thickness=self.thickness)
                cv2.line(inst_img, (x[i - 1], y[i - 1]), (x[i], y[i]),
                         (lane_id, lane_id, lane_id), thickness=self.thickness)

        self.signal_save_finished.emit(sem_seg_img, inst_img)

    def _process_image(self, image, colors):
        predictions_numpy = self._process_layer_image(image, colors)
        if predictions_numpy is None:
            result = self.original_image
        else:
            result = cv2.addWeighted(self.original_image, 0.8, predictions_numpy, 0.5, 0.0)
        return result

    def _process_layer_image(self, image, colors):
        if np.all(image == 0) or image is None:
            return self.original_image

        if self.original_image is None:
            self.signal_error_signal.emit("Data NONE!", self.filter_id)
            return None

        predictions_numpy = image.astype(np.uint8)

        if len(colors) < int(np.max(predictions_numpy)):
            self.signal_error_signal.emit("Segmentation image corrupted!", self.filter_id)
            return None

        if predictions_numpy.shape[-1] == 3:
            predictions_numpy = cv2.cvtColor(predictions_numpy, cv2.COLOR_BGR2GRAY)

        predictions_numpy = colors[predictions_numpy]
        predictions_numpy = np.array(predictions_numpy, dtype='uint8')
        predictions_numpy = cv2.resize(predictions_numpy, (self.original_image.shape[1], self.original_image.shape[0]))

        return predictions_numpy

    # Private Methods - Thetas -----------------------------------------------------------------------------------------
    def _points_to_model(self, lane_id):
        if self.cam is None:
            self.signal_error_signal.emit("Camera NONE!", self.filter_id)
            return

        if lane_id not in self.laneObjects.keys():
            self.signal_error_signal.emit("Lane id {} not found!".format(lane_id), self.filter_id)
            return

        lane_pts_reproj_3d_cam = []
        for i, point in enumerate(self.laneObjects[lane_id]):
            if point is not None:
                x = int(point.pos().x()) + self.pen_size // 2
                y = self._get_line_h(i) + self.pen_size // 2
                lane_pts_reproj_3d_cam.append(lane_3d_from_2d(self.cam, x, y))

        lane_pts_reproj_3d_cam = np.array(lane_pts_reproj_3d_cam)

        lane_pts_reproj_3d_veh = list()
        for idx in range(len(lane_pts_reproj_3d_cam)):
            pt_3d_veh = np.matmul(self.cam.T_cam2body, lane_pts_reproj_3d_cam[idx])
            lane_pts_reproj_3d_veh.append(pt_3d_veh)

        lane_pts_reproj_3d_veh = np.array(lane_pts_reproj_3d_veh)
        x = lane_pts_reproj_3d_veh[:, 0]
        y = lane_pts_reproj_3d_veh[:, 1]
        lane_model = np.flip(np.polyfit(x, y, 3))

        self.thetas[lane_id] = [round(lane_model[0], 5), round(lane_model[1], 5),
                                round(lane_model[2], 5), round(lane_model[3], 5)]

    def _model_to_image(self, lane_id):
        """
        Use model (theta_0, theta_1, theta_2, theta_3) to generate points in image domain
        """
        if self.cam is None:
            self.signal_error_signal.emit("Camera NONE!", self.filter_id)
            return []

        if lane_id not in self.laneObjects.keys():
            self.signal_error_signal.emit("Lane id {} not found!".format(lane_id), self.filter_id)
            return []

        # Generate lane in vehicle coordinates
        lane_pts_synth_3d_veh = generate_lane(self.thetas[lane_id][0],
                                              self.thetas[lane_id][1],
                                              self.thetas[lane_id][2],
                                              self.thetas[lane_id][3])

        # Project synthetic lane to image
        lane_pts_synth_3d_cam = []
        for i in range(len(lane_pts_synth_3d_veh)):
            pt_3d_cam = np.matmul(self.cam.T_body2cam, lane_pts_synth_3d_veh[i])
            lane_pts_synth_3d_cam.append(pt_3d_cam)
        lane_pts_synth_3d_cam = np.array(lane_pts_synth_3d_cam)

        # Project synthetic lane points into image
        lane_pts_2d_cam = self.cam.world2sensor(lane_pts_synth_3d_cam)
        return lane_pts_2d_cam

    def _draw_theta_lanes(self):
        """
        Draw lanes from parameters: theta_0, theta_1, theta_2, theta_3
        Display image in third tab (Thetas img)
        """
        image = self.original_image.copy()
        color = self.object_classes.colormap()

        for lane_id in self.thetas.keys():
            pts_2d = self._model_to_image(lane_id)

            for i in range(1, len(pts_2d)):
                try:
                    cv2.line(image, (int(pts_2d[i-1][0]), int(pts_2d[i-1][1])), (int(pts_2d[i][0]), int(pts_2d[i][1])),
                             (int(color[lane_id][0]), int(color[lane_id][1]), int(color[lane_id][2])),
                             thickness=self.thickness)
                except TypeError:
                    print(lane_id, color[lane_id][0], color[lane_id][1], color[lane_id][2])
                    raise
        self.load_image_in_view(image, self.qtCamThetas)

    # Private Methods - Save data --------------------------------------------------------------------------------------
    def _save_frame_based_data(self, base_path):
        file_path = os.path.join(base_path, 'framebased_data_descriptor.csv')
        file_copy_path = os.path.join(base_path, 'framebased_data_descriptor_copy.csv')
        shutil.copyfile(file_path, file_copy_path)
        wrote = False
        lines = [self.frame_based_header]

        if os.path.exists(file_path):
            with open(file_path, "r") as f_in:
                lines = f_in.readlines()

        with open(file_path, "w") as f_out:
            for line in lines:
                if line == self.frame_based_header:
                    f_out.write(self.frame_based_header)
                else:
                    line_timestamp_stop = int(line.split(",")[0])
                    current_ts_stop = int(self.qtTsStop.text())
                    if line_timestamp_stop == current_ts_stop:
                        if wrote is False:
                            wrote = True
                            for index in self.laneObjects.keys():
                                # Lane
                                lane = []
                                for item in self.laneObjects[index]:
                                    if item is None:
                                        lane.append(-1)
                                    else:
                                        if int(item.pos().x()) != -1:
                                            lane.append(int(item.pos().x()))
                                        else:
                                            lane.append(int(0))

                                # Thetas
                                thetas = ',,,'
                                if index in self.thetas.keys():
                                    if self.thetas[index] is not None:
                                        thetas = '{},{},{},{}'.format(self.thetas[index][0], self.thetas[index][1],
                                                                      self.thetas[index][2], self.thetas[index][3])
                                f_out.write("{},{},{},{}\n".format(
                                    current_ts_stop, index, str(lane).replace(",", ""), thetas))
                    else:
                        f_out.write(line)
            if wrote is False:
                for index in self.laneObjects.keys():
                    # Lanes
                    lane = []
                    for item in self.laneObjects[index]:
                        if item is None:
                            lane.append(-1)
                        else:
                            if int(item.pos().x()) != -1:
                                lane.append(int(item.pos().x()))
                            else:
                                lane.append(int(0))

                    # Thetas
                    thetas = ',,,'
                    if index in self.thetas.keys():
                        if self.thetas[index] is not None:
                            thetas = '{},{},{},{}'.format(self.thetas[index][0], self.thetas[index][1],
                                                          self.thetas[index][2], self.thetas[index][3])
                    f_out.write("{},{},{},{}\n".format(
                        int(self.qtTsStop.text()), index, str(lane).replace(",", ""), thetas))
        os.remove(file_copy_path)

    def _save_data_descriptor(self, sem_seg_img, inst_img):
        base_path = self.parser.get_parser_base_path(self.filter_id)
        file_path = os.path.join(base_path, 'data_descriptor.csv')
        semseg_path = "samples/0/left"
        inst_path = "samples/1/left"
        wrote = False

        with open(file_path, "r") as f_in:
            lines = f_in.readlines()

        with open(file_path, "w") as f_out:
            for line in lines:
                if line == "timestamp_stop,sampling_time,semantic,instances\n":
                    f_out.write(line)
                else:
                    line_timestamp_stop = line.split(",")[0]
                    line_sampling_time = line.split(",")[1]

                    if line.split(",")[2] != "-1" and len(line.split(",")[2]) > 0:
                        semseg_path = os.path.dirname(line.split(",")[2])

                    if line.split(",")[3] != "\n":
                        inst_path = os.path.dirname(line.split(",")[3].replace('\n', ""))

                    if line_timestamp_stop == self.qtTsStop.text():
                        if wrote is False:
                            wrote = True
                            f_out.write("{},{},{}/{}.png,{}/{}.png\n".format(self.qtTsStop.text(),
                                                                             self.qtTsSamplingTime.text(),
                                                                             semseg_path,
                                                                             self.qtTsStop.text(),
                                                                             inst_path,
                                                                             self.qtTsStop.text()))
                    else:
                        f_out.write(line)

        cv2.imwrite("{}/{}/{}.png".format(base_path, semseg_path, self.qtTsStop.text()), sem_seg_img)
        cv2.imwrite("{}/{}/{}.png".format(base_path, inst_path, self.qtTsStop.text()), inst_img)

    # Slots Buttons ----------------------------------------------------------------------------------------------------
    def on_click_button_clear_frame(self):
        # Clear points
        for lane in self.laneObjects.values():
            for point in lane:
                if point is not None:
                    self.scene.removeItem(point)
        self.laneObjects.clear()

        # Thetas
        self.thetas.clear()

    def on_click_button_copy_frame(self):
        self.frame_clipboard.clear()
        self.frame_clipboard = {}
        for lane_id in self.laneObjects.keys():
            lane = []
            for item in self.laneObjects[lane_id]:
                if item is None:
                    lane.append(-1)
                else:
                    lane.append(int(item.pos().x()))
            self.frame_clipboard[lane_id] = lane

    def on_click_button_paste_frame(self):
        if len(self.laneObjects.keys()) != 0:
            return

        self._load_lanes(self.frame_clipboard)
        self.update_save_button(True)

    # Slots Points -----------------------------------------------------------------------------------------------------
    def on_draw_new_point(self, x, y):
        point = self._draw_point(x, self._get_line_h(self.height_index))
        self.scene.addItem(point)

        if self.laneObjects[self.last_item_id][self.height_index] is not None:
            self._delete_point()

        self.laneObjects[self.last_item_id][self.height_index] = point
        self.update_save_button(True)

    def on_delete_point(self):
        self._delete_point()

    def on_move_temp_point(self, x, y):
        if self.temp_point is not None:
            self.temp_point.setX(x)
            self.temp_point.setY(self._get_line_h(self.height_index))

    def on_selection_changed(self):
        try:
            if not self.draw_enabled:
                self.show_item_id()
                self.selected_item = None
                if len(self.scene.selectedItems()) == 0:
                    return

                self.selected_item = self.scene.selectedItems()[0]
                self.show_item_id(self.selected_item.get_item_id())
        except RuntimeError:
            pass

    # Slots ------------------------------------------------------------------------------------------------------------
    def on_save_finished(self, sem_seg_img, inst_img):
        self._save_data_descriptor(sem_seg_img, inst_img)
        color = self.object_classes.colormap()
        overlay_sem_seg_img = self._process_image(sem_seg_img, color)
        overlay_inst_img = self._process_image(inst_img, color)
        self.load_image_in_view(overlay_sem_seg_img, self.qtCamSegmented)
        self.load_image_in_view(overlay_inst_img, self.qtCamInstances)
        QApplication.processEvents()

    def on_click_generate_thetas(self):
        self.thetas.clear()
        try:
            for lane_id in self.laneObjects.keys():
                self._points_to_model(lane_id)
            self._draw_theta_lanes()
            self.update_save_button(True)
        except np.linalg.LinAlgError:
            self.signal_error_signal.emit("Invalid calibration file for calculating thetas in image filter id {}!"
                                          .format(self.filter_id_image), self.filter_id)
            return

    def on_change_check_box(self):
        if self.qtHBox.isChecked():
            self._show_lines(True)
        else:
            self._show_lines(False)

    # Inherited Slots Load Data ----------------------------------------------------------------------------------------
    def on_load_data(self, line_index, ts_sync_timestamp):
        self.clear()

        self.show_runtime_values(frame_id=line_index - 1)
        self.show_item_id()
        data, img = super().on_load_data(line_index, ts_sync_timestamp)

        if img is None or data is None:
            return

        self.original_image = img[3]

        if self.original_image is not None:
            self.pen_size = max(self.original_image.shape[1], self.original_image.shape[0]) // default_pen_size_ratio
            self.thickness = max(self.original_image.shape[1], self.original_image.shape[0]) // default_thickness_ratio

        color = self.object_classes.colormap()
        ts_stop, sampling_time, ts_image, cv_image, _ = img
        self.show_runtime_values(ts_stop=ts_stop,
                                 ts_sampling_time=sampling_time,
                                 frame_id=line_index - 1,
                                 ts_image=ts_image)
        self.load_image(cv_image)
        self.qtHBox.setChecked(True)
        self._create_lines()
        self._create_temp_point()
        _, _, seg_img, inst_img, points, thetas = data
        points_loaded = self._load_lanes(points)

        self.thetas = thetas
        self._draw_theta_lanes()

        seg_img_cam = self._process_image(seg_img, color)
        self.load_image_in_view(seg_img_cam, self.qtCamSegmented)
        inst_img_cam = self._process_image(inst_img, color)
        self.load_image_in_view(inst_img_cam, self.qtCamInstances)

        self.on_main_splitter_moved()
        self.show_number_of_items(len(self.laneObjects.keys()))
        self.qtItemID.setText(str(-1))
        self.update_save_button(False)

        # Inherited Slots Widget -------------------------------------------------------------------------------------------

    # Slots Draw/Delete/Save -------------------------------------------------------------------------------------------
    def on_click_button_draw(self):
        if self.draw_enabled:
            self.draw_enabled = False
            self.signal_activate_draw.emit()
            self.qtButtonDrawLane.setText("Draw Lane")
            self.temp_point.hide()
            self.selected_item = None
            if not self._check_lane(self.last_item_id):
                self._delete_lane(self.last_item_id)
            self.qtItemID.setText(str(-1))
            self.show_number_of_items(len(self.laneObjects.keys()))
            self.show_item_id()
        else:
            self.draw_enabled = True
            self.signal_activate_draw.emit()
            self.qtButtonDrawLane.setText("Stop Drawing Lane")
            self.temp_point.show()

            if self.selected_item is not None:
                self.last_item_id = self.selected_item.get_item_id()
                if len(self.scene.selectedItems()):
                    self.scene.selectedItems()[0].setSelected(False)
            else:
                if len(self.laneObjects) != 0:
                    self.last_item_id = max(self.laneObjects.keys()) + 1
                else:
                    self.last_item_id = 1
                self.laneObjects[self.last_item_id] = []
                for _ in range(len(self.h_lines)):
                    self.laneObjects[self.last_item_id].append(None)

                self.show_item_id(self.last_item_id)

    def on_click_button_delete(self):
        if len(self.scene.selectedItems()) > 0:
            lane_id = self.selected_item.get_item_id()

            self._delete_lane(lane_id)
            self.show_number_of_items(len(self.laneObjects.keys()))
            self.qtItemID.setText(str(-1))

    def on_click_button_save(self):
        base_path = self.parser.get_parser_base_path(self.filter_id)
        self._save_frame_based_data(base_path)

        self._generate_images()

        self.update_save_button(False)
        if self.draw_enabled:
            self.on_click_button_draw()
        self.on_main_splitter_moved()
        self.qtItemID.setText(str(-1))

    # Inherited Slots --------------------------------------------------------------------------------------------------
    def on_main_splitter_moved(self):
        if self.original_image is not None:
            self.scene.setSceneRect(0, 0, self.original_image.shape[1], self.original_image.shape[0])
        self.qtGraphicsView.fitInView(self.qtGraphicsView.scene().sceneRect(), Qt.KeepAspectRatio)
        self.qtCamSegmented.fitInView(self.qtCamSegmented.scene().sceneRect(), Qt.KeepAspectRatio)
        self.qtCamInstances.fitInView(self.qtCamInstances.scene().sceneRect(), Qt.KeepAspectRatio)
        self.qtCamThetas.fitInView(self.qtCamThetas.scene().sceneRect(), Qt.KeepAspectRatio)

    # Inherited Slots Copy/Cut/Paste/Undo ------------------------------------------------------------------------------
    def on_copy(self):
        if len(self.scene.selectedItems()) > 0:
            lane_id = self.selected_item.get_item_id()
            self.clipboard = ["copy", self.get_item_details(lane_id)]

    def on_cut(self):
        if len(self.scene.selectedItems()) > 0:
            lane_id = self.selected_item.get_item_id()
            self.clipboard = ["cut", self.get_item_details(lane_id)]
            self.on_click_button_delete()
            self.update_save_button(True)

    def on_paste(self):
        if len(self.scene.selectedItems()) != 0:
            for lane in self.scene.selectedItems():
                lane.setSelected(False)

        if self.clipboard is None:
            return

        # Extract info
        lane = self.clipboard[1].replace('\n', '').split(',')
        timestamp_stop = lane[0]
        lane_id = int(lane[1])
        lane_p = lane[2]

        # Calculate lane_id
        self.last_item_id = lane_id
        if len(self.laneObjects) != 0:
            if lane_id in self.laneObjects.keys():
                self.last_item_id = max(self.laneObjects.keys()) + 1

        # Create object in dict
        self.laneObjects[self.last_item_id] = []
        for _ in range(len(self.h_lines)):
            self.laneObjects[self.last_item_id].append(None)

        # Prepare points string
        points_str = lane_p[1:-1]
        points_str = points_str.split(" ")

        # Add points to dict and to scene
        for i, point in enumerate(points_str):
            if point != '-1':
                qt_point = self._draw_point(int(point), self._get_line_h(i))
                self.laneObjects[self.last_item_id][i] = qt_point
                self.scene.addItem(qt_point)

        # self.clipboard = None
        self.show_number_of_items(len(self.laneObjects.keys()))
        self.update_save_button(True)
