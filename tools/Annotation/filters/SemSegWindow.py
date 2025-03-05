"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from tools.Annotation.filters.FilterWindowInterface import FilterWindowInterface, ChangesItem
from tools.Annotation.items.DojoGraphicScene import DojoGraphicScene
from tools.Annotation.items.SemSegPixmapItem import SemSegPixmapItem
from tools.Annotation.items.SemSegPointItem import SemSegPointItem
from tools.Annotation.items.SemSegItem import SemSegItem

from toolkit.env.object_classes import ObjectClasses, get_countable_classes

from PyQt5 import uic, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import numpy as np
import shutil
import cv2
import os


current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")


class SemSegWindow(FilterWindowInterface):
    # Signals ----------------------------------------------------------------------------------------------------------
    signal_class_selected = pyqtSignal()
    signal_save_finished = pyqtSignal(object, object)

    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id, input_sources=None):
        super().__init__(filter_id, input_sources)
        uic.loadUi(os.path.join(ui_directory, 'SemSegWindow.ui'), self)
        self.setWindowTitle("Semantic Segmentation Viewer")

        # Widget - children
        self.find_children()

        # Widget - Buttons
        self.qtCamSegmented = self.findChild(QGraphicsView, 'qtCamSegmented')
        self.qtCamInstances = self.findChild(QGraphicsView, 'qtCamInstances')
        self.qtButtonDrawShape = self.findChild(QPushButton, 'qtButtonDrawShape')
        self.qtButtonDeleteShape = self.findChild(QPushButton, 'qtButtonDeleteShape')
        self.qtButtonBringToFront = self.findChild(QPushButton, 'qtButtonBringToFront')
        self.qtButtonSendToBack = self.findChild(QPushButton, "qtButtonSendToBack")
        self.qtButtonCopyFrame = self.findChild(QPushButton, 'qtButtonCopyFrame')
        self.qtButtonPasteFrame = self.findChild(QPushButton, "qtButtonPasteFrame")
        self.qtButtonClearFrame = self.findChild(QPushButton, "qtButtonClearFrame")

        # Widget
        self.qtSegSplitter = self.findChild(QSplitter, 'qtSegSplitter')
        self.qtOpacitySlider = self.findChild(QSlider, 'qtOpacitySlider')
        self.qtInstanceId = self.findChild(QSpinBox, 'qtInstanceID')
        self.qtTabs = self.findChild(QTabWidget, 'tabWidget')

        # Widget - Tabs
        self.qtSegmentedTabOriginalImage = self.findChild(QGraphicsView, 'qtSegmentedTabOriginalImage')
        self.qtSegmentedTabSemanticImage = self.findChild(QGraphicsView, 'qtSegmentedTabSemanticImage')
        self.qtSegmentedTabInstanceImage = self.findChild(QGraphicsView, 'qtSegmentedTabInstanceImage')

        # Members
        self.object_classes = ObjectClasses(self.object_classes_path)
        self.countable_classes = get_countable_classes(self.object_classes_path)
        self.scene = DojoGraphicScene()
        self.frame_based_header = "timestamp_stop,shape_id,cls,instance,points\n"
        self.point_brush = QBrush(QColor(255, 255, 255, 255))
        self.brush = QBrush(QColor(255, 255, 255, 255))
        self.point_pen = QPen(QColor(0, 0, 0), 5)
        self.pen = QPen(QColor(0, 0, 0), 5)
        self.original_image = None
        self.zoom_enabled = True
        self.draw_class = None
        self.new_shape = True
        self.shape_moved_last_position = None
        self.shape_move_draw = False
        self.segmentedObjects = {}
        self.copy_shapes_frame = []
        self.points = []
        self.points_dict = {}
        self.radius = 5
        self.point_index = 0
        self.opacity = 0.5
        self.zoom = 1

        # Initialization
        self.init_connections()
        self.init_tree_widget()
        self.init_combo_overlay()
        self.init_window()
        self.init_splitter()

    # Events / Slots ---------------------------------------------------------------------------------------------------
    def on_mouse_release_event(self):
        if self.shape_moved_last_position is None:
            return

        shape = None
        if self.scene and len(self.scene.selectedItems()) > 0:
            shape = self.scene.selectedItems()[0]

        if shape is None:
            return

        if shape.__class__ == SemSegPointItem:
            return

        distance_x = shape.pos().x() - self.shape_moved_last_position.x()
        distance_y = shape.pos().y() - self.shape_moved_last_position.y()
        self.shape_move_draw = True

        for key, value in self.points_dict.items():
            for point in value:
                if point is None:
                    continue
                if key == self.last_item_id:
                    point.moveBy(distance_x, distance_y)

        self.shape_move_draw = False
        self.shape_moved_last_position = None
        shape.moveBy(-distance_x, -distance_y)
        self._draw_polygon(self.last_item_id)
        shape.move_finished()
        self._show_points(self.last_item_id, True)

    # Inherited Initialization Methods ---------------------------------------------------------------------------------
    def init_window(self):
        self.image_timestamps = self.parser.get_all_timestamps(filter_id=self.filter_id_image)
        self.synced_timestamps = self.parser.get_records_synced(filter_ids=[self.filter_id_image, self.filter_id])

        self.qtGraphicsView.setScene(self.scene)
        self.qtCamSegmented.setScene(QGraphicsScene())
        self.qtCamInstances.setScene(QGraphicsScene())
        self.update_save_button(False)

        self.qtWarning.setStyleSheet("QLabel {background-color:red}")
        self.show_warning_text(False)

        self.qtSegmentedTabOriginalImage.setScene(QGraphicsScene())
        self.qtSegmentedTabSemanticImage.setScene(QGraphicsScene())
        self.qtSegmentedTabInstanceImage.setScene(QGraphicsScene())

        self.qtOpacitySlider.setValue(int(self.opacity*100))
        self.on_main_splitter_moved()

    def init_connections(self):
        # Scene
        self.scene.selectionChanged.connect(self.on_highlighted_class)
        self.scene.signal_draw_new_point.connect(self.on_draw_new_point)
        self.scene.signal_shape_moved.connect(self.on_mouse_release_event)

        # Widgets
        self.splitter.splitterMoved.connect(self.on_main_splitter_moved)
        self.qtSegSplitter.splitterMoved.connect(self.on_main_splitter_moved)
        self.qtTabs.currentChanged.connect(self.on_main_splitter_moved)
        self.qtComboOverlay.currentTextChanged.connect(self.on_change_text_combo)
        self.qtOpacitySlider.valueChanged.connect(self.on_opacity_changed)
        self.qtInstanceId.valueChanged.connect(self.on_instance_id_change)
        self.qtClasses.itemClicked.connect(self.on_select_class)

        # Buttons
        self.qtButtonDrawShape.clicked.connect(self.on_click_button_draw)
        self.qtButtonSave.clicked.connect(self.on_click_button_save)
        self.qtButtonBringToFront.clicked.connect(self.on_click_bring_to_front)
        self.qtButtonSendToBack.clicked.connect(self.on_click_send_to_back)
        self.qtButtonDeleteShape.clicked.connect(self.on_click_button_delete)
        self.qtButtonCopyFrame.clicked.connect(self.on_click_copy_frame)
        self.qtButtonPasteFrame.clicked.connect(self.on_click_paste_frame)
        self.qtButtonClearFrame.clicked.connect(self.on_click_clear_frame)

        # Signals
        self.signal_save_finished.connect(self.on_save_finished)
        self.signal_class_selected.connect(self.on_change_class_item)
        self.signal_activate_draw.connect(self.scene.on_active_draw)

    # Inherited Methods ------------------------------------------------------------------------------------------------
    def clear(self):
        if self.qtButtonSave.isEnabled():
            self.on_click_button_save()

        # Annotation Tab
        if self.draw_enabled:
            self.on_click_button_draw()
        self.update_save_button(False)

        if len(self.segmentedObjects) != 0:
            if self.qtButtonSave.isEnabled():
                self.on_click_button_save()

        self.segmentedObjects.clear()
        self.points_dict.clear()
        self.scene.clear()

        # Instances Page
        self.qtCamSegmented.scene().clear()
        self.qtCamInstances.scene().clear()
        self.qtSegmentedTabOriginalImage.scene().clear()
        self.qtSegmentedTabSemanticImage.scene().clear()
        self.qtSegmentedTabInstanceImage.scene().clear()

        self.refresh()

    def refresh(self):
        self.qtNoObjects.setText(str(len(self.segmentedObjects)))

    def get_item_details(self, shape):
        return "{},{},{},{},{}\n".format(self.qtTsStop.text(),
                                         int(shape.get_item_id()),
                                         int(shape.get_class()),
                                         int(shape.instance_id),
                                         shape.get_points())

    # Static Methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _distance_point_point(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    # Methods General --------------------------------------------------------------------------------------------------
    def _redraw(self, shapes):
        for shape_old in self.segmentedObjects.values():
            self.scene.removeItem(shape_old)

        self.segmentedObjects.clear()
        self._load_shapes(shapes)
        self.update_save_button(True)

    def _is_countable(self, item_class):
        return self.object_classes.is_countable(item_class)

    # Private Methods Images -------------------------------------------------------------------------------------------
    def _load_annotation_image(self):
        if self.original_image is None:
            self.signal_error_signal.emit("Image is None!", self.filter_id)
            return

        qtImage = QtGui.QImage(self.original_image, self.original_image.shape[1], self.original_image.shape[0],
                               self.original_image.shape[1] * 3, QtGui.QImage.Format_RGB888)
        canvas = QPixmap(self.original_image.shape[1], self.original_image.shape[0])
        canvas.fill(QColor(0, 0, 0))
        self.scene.addPixmap(canvas)
        image = QPixmap(qtImage)
        self.scene.addPixmap(image)

    def _load_processed_image_in_view(self, cv_image, view, mask=None):
        if cv_image is None:
            self.signal_error_signal.emit("Data is none! Filter id {}!".format(self.frame_id), self.filter_id)
            return

        image = QtGui.QImage(cv_image, cv_image.shape[1], cv_image.shape[0], cv_image.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        image = QPixmap(image)

        qt_pixmap = SemSegPixmapItem(image, self.object_classes, mask)
        view.scene().addItem(qt_pixmap)
        view.fitInView(view.scene().sceneRect(), Qt.KeepAspectRatio)

    def _process_image(self, image, colors):
        predictions_numpy = self._process_layer_image(image, colors)
        if predictions_numpy is None:
            result = self.original_image
        else:
            result = cv2.addWeighted(self.original_image, 0.8, predictions_numpy, 0.5, 0.0)
        return result

    def _process_layer_image(self, image, colors):
        if np.all(image == 0):
            return self.original_image

        if self.original_image is None:
            self.signal_error_signal.emit("Data is none! Frame id {}!".format(self.frame_id), self.filter_id)
            return None

        predictions_numpy = image.astype(np.uint8)

        if len(colors) < int(np.max(predictions_numpy)):
            self.signal_error_signal.emit(
                "Corrupt segmentation image! Frame id {}!".format(self.frame_id), self.filter_id)
            return None

        if predictions_numpy.shape[-1] == 3:
            predictions_numpy = cv2.cvtColor(predictions_numpy, cv2.COLOR_BGR2GRAY)

        predictions_numpy = colors[predictions_numpy]
        predictions_numpy = np.array(predictions_numpy, dtype='uint8')
        predictions_numpy = cv2.resize(predictions_numpy, (self.original_image.shape[1], self.original_image.shape[0]))

        return predictions_numpy

    def _fill_segmented_layers(self, sem_seg, inst):
        self.load_image_in_view(self.original_image, self.qtSegmentedTabOriginalImage)

        if inst is not None:
            self._load_processed_image_in_view(inst, self.qtSegmentedTabInstanceImage)

        if sem_seg is not None:
            self._load_processed_image_in_view(sem_seg, self.qtSegmentedTabSemanticImage)

    # Private Methods Shapes -------------------------------------------------------------------------------------------
    def _load_shapes(self, shapes):
        self.show_number_of_items(len(shapes))
        for shape in shapes:
            if self.last_item_id in self.segmentedObjects:
                self.segmentedObjects[self.last_item_id].setSelected(False)

            self.last_item_id = int(shape[1])
            shape_cls = int(shape[2])

            if self.object_classes.get_class_max_index() < shape_cls:
                self.qtWarning.setText("ATTENTION: Invalid object classes file!")
                self.signal_error_signal.emit("Invalid object class file!", self.frame_id)
                self.show_warning_text(True)
                continue

            shape_pts = shape[-1]
            shape_instance = int(shape[3])
            self._draw_shape(shape_cls, shape_instance, shape_pts)
        self.refresh()

    def _draw_shape(self, class_index, shape_instance=0, points=None, position=-1):
        item = self.get_class_item(class_index)

        if item is not None:
            self.brush = item.background(1)
        else:
            self.brush = QBrush(Qt.white)

        class_name = self.object_classes.get_name_by_index(class_index)
        segmented_object = SemSegItem(self.last_item_id, [class_index, class_name],
                                      self.brush, self.pen)
        segmented_object.setBrush(self.brush)
        segmented_object.setOpacity(self.opacity)
        segmented_object.instance_id = shape_instance
        segmented_object.signals.signal_item_changed.connect(self.on_shape_moved)
        self.points_dict[self.last_item_id] = list()
        segmented_object.setSelected(True)

        if points is not None:
            segmented_object.set_points(points)

        if position != -1:
            items = list(self.segmentedObjects.items())
            items.insert(position, (self.last_item_id, segmented_object))
            shapes = []

            for shape_old in dict(items).values():
                shape_info = [self.frame_id,
                              shape_old.item_id,
                              shape_old.get_class(),
                              shape_old.instance_id,
                              [(p.x(), p.y()) for p in shape_old.polygon()]]
                shapes.append(shape_info)

            self._redraw(shapes)
        else:
            self.segmentedObjects[self.last_item_id] = segmented_object
            self.scene.addItem(segmented_object)

        if points is not None:
            i = 0
            for p in points:
                point = SemSegPointItem(self.last_item_id, i, self.point_pen, self.point_brush,
                                        p[0], p[1], 0, 0, self.radius, self.radius)
                point.set_as_shape_point()
                point.set_connection(self.on_point_changed, self.on_edit_finish)
                self.points_dict[self.last_item_id].append(point)
                self.scene.addItem(point)
                i += 1
        self.qtNoObjects.setText(str(len(self.segmentedObjects)))
        segmented_object.setSelected(False)
        self.refresh()
        return segmented_object

    def _delete_shape(self, shape):
        self.changes.add_item(
            ChangesItem("{},{}".format(self.get_item_details(self.segmentedObjects[self.last_item_id]),
                                       list(self.segmentedObjects.keys()).index(self.last_item_id)), was_removed=True))
        self.update_save_button(True)
        self.segmentedObjects.pop(shape.get_item_id())
        self.scene.removeItem(shape)
        self.qtNoObjects.setText(str(len(self.segmentedObjects)))
        for point in self.points_dict[shape.get_item_id()]:
            self.scene.removeItem(point)
        self.points_dict.pop(shape.get_item_id())
        self.qtItemID.setText(str(-1))
        self.refresh()

    # Private Methods Polygons -----------------------------------------------------------------------------------------
    def _draw_polygon(self, index):
        if self.shape_move_draw:
            return

        polygon = []
        for p in self.points_dict[index]:
            polygon.append([p.pos().x(), p.pos().y()])

        self.segmentedObjects[index].set_points(polygon)

    def _update_polygon(self, class_index, points = None):
        item = self.get_class_item(class_index)
        self.segmentedObjects[self.last_item_id].setBrush(self.brush)
        self.segmentedObjects[self.last_item_id].set_class(class_index, item.background(1))

        if points is not None:
            for point in self.points_dict[self.last_item_id]:
                self.scene.removeItem(point)

            self.points_dict[self.last_item_id] = list()
            i = 0

            for p in points:
                point = SemSegPointItem(self.last_item_id, i, self.point_pen, self.point_brush,
                                        p[0], p[1], 0, 0, self.radius, self.radius)
                point.set_as_shape_point()
                point.set_connection(self.on_point_changed, self.on_edit_finish)
                self.points_dict[self.last_item_id].append(point)
                self.scene.addItem(point)
                i += 1

            self.segmentedObjects[self.last_item_id].set_points(points)

    # Private Methods Points -------------------------------------------------------------------------------------------
    def _show_points(self, index, show=False):
        try:
            for key, value in self.points_dict.items():
                for point in value:
                    if point is None:
                        continue
                    if show and key == index:
                        point.show()
                    else:
                        point.hide()
        except RuntimeError:
            self.signal_error_signal.emit("RuntimeError - _show_points({},{})".format(index, show), self.frame_id)

    def _delete_point(self, point):
        self.changes.add_item(ChangesItem(self.get_item_details(self.segmentedObjects[self.last_item_id])))
        self.scene.removeItem(point)
        self.points_dict[self.last_item_id].remove(point)
        self._draw_polygon(self.last_item_id)
        self.segmentedObjects[self.last_item_id].setSelected(True)
        self.update_save_button(True)
        self.refresh()

    # Private Methods Save ---------------------------------------------------------------------------------------------
    def _generate_images(self):
        org_img = self.original_image[..., ::-1].copy()
        sem_seg_img = np.zeros((org_img.shape[0], org_img.shape[1], 3), dtype=np.uint8)
        inst_img = np.zeros((org_img.shape[0], org_img.shape[1], 3), dtype=np.uint8)

        for index, shape in self.segmentedObjects.items():
            contour = list()
            for pt in self.points_dict[index]:
                pt_x = pt.x()
                pt_y = pt.y()
                if pt_x < 0:
                    pt_x = 0
                elif pt_x > org_img.shape[1]:
                    pt_x = org_img.shape[1]
                if pt_y < 0:
                    pt_y = 0
                elif pt_y > org_img.shape[0]:
                    pt_y = org_img.shape[0]
                contour.append([pt_x, pt_y])
            contour = np.array(contour, dtype="int32")
            cv2.fillPoly(sem_seg_img, [contour], (shape.class_, shape.class_, shape.class_))
            cv2.fillPoly(inst_img, [contour], (shape.instance_id, shape.instance_id, shape.instance_id))

        self.signal_save_finished.emit(sem_seg_img, inst_img)

    def _save_image(self):
        self.on_main_splitter_moved()
        if self.last_item_id in self.segmentedObjects.keys():
            self.segmentedObjects[self.last_item_id].setSelected(False)
        self._show_points(-1)
        self._generate_images()

    def _save_frame_based_data(self, base_path):
        file_path = os.path.join(base_path, 'framebased_data_descriptor.csv')
        file_copy_path = os.path.join(base_path, 'framebased_data_descriptor_copy.csv')
        shutil.copyfile(file_path, file_copy_path)
        wrote = False
        lines = [self.frame_based_header]
        invalid_shapes = []

        if os.path.exists(file_path):
            with open(file_path, "r") as f_in:
                lines = f_in.readlines()

        with open(file_path, "w") as f_out:
            for line in lines:
                if line == "timestamp_stop,shape_id,cls,points\n" or line == self.frame_based_header:
                    f_out.write(self.frame_based_header)
                else:
                    line_timestamp_stop = int(line.split(",")[0])
                    current_ts_stop = int(self.qtTsStop.text())
                    if line_timestamp_stop == current_ts_stop:
                        if wrote is False:
                            wrote = True
                            for index, shape in self.segmentedObjects.items():
                                if shape.polygon() is None or len(shape.polygon()) < 3:
                                    invalid_shapes.append(shape)
                                    continue
                                f_out.write(self.get_item_details(shape))
                    else:
                        f_out.write(line)
            if wrote is False:
                for index, shape in self.segmentedObjects.items():
                    f_out.write(self.get_item_details(shape))

        os.remove(file_copy_path)
        for shape in invalid_shapes:
            self._delete_shape(shape)

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

    # Slots ------------------------------------------------------------------------------------------------------------
    def on_opacity_changed(self, value):
        self.show_warning_text(False)
        self.opacity = value / 100

        if self.opacity < 0.2:
            self.signal_warning_signal.emit("Opacity value is too low!", self.filter_id)
            self.qtWarning.setText("ATTENTION: Opacity is {}, too low!".format(self.opacity))
            self.show_warning_text(True)

        for key, shape in self.segmentedObjects.items():
            shape.setOpacity(self.opacity)

    def on_shape_moved(self, x, y, w, h):
        if self.shape_moved_last_position is not None:
            return

        self.changes.add_item(ChangesItem(self.get_item_details(self.segmentedObjects[self.last_item_id])))
        self.shape_moved_last_position = QPointF(x, y)
        self._show_points(-1)

    def on_point_changed(self, x, y, z, h):
        self.update_save_button(True)
        if len(self.scene.selectedItems()) == 0:
            return

        point = self.scene.selectedItems()[0]
        index = point.get_item_id()
        self._draw_polygon(index)

        if self.edit_finish:
            self.changes.add_item(ChangesItem(self.get_item_details(self.segmentedObjects[index])))
            self.edit_finish = False

    def distance_line_point(self, p1, p2, target):
        """
        Description:
            Method calculates the distance between target's coords and the line made by
            the two points p1 and p2.
            Source: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
        Parameters:
            p1: First point of the line
            p2: Second point of the line
            target: Target point
        Returns:
            float: the dist between the line p1-p2 and the point target
        """
        p12_dist = self._distance_point_point(p1, p2)
        return abs((p2[0]-p1[0])*(p1[1]-target[1]) - (p1[0]-target[0])*(p2[1]-p1[1])) / p12_dist

    def on_draw_new_point(self, x, y):
        if self.last_item_id not in self.points_dict.keys():
            self.signal_warning_signal.emit("Shape ID {} not found!"
                                            .format(self.last_item_id), self.filter_id)
            return

        idx = 0
        if self.new_shape:
            self.points.append([x, y])
        else:
            target_point = [x, y]
            idx = 0
            min_dist = self.distance_line_point(self.points[0], self.points[-1], target_point)
            entered = False
            for i in range(0, len(self.points)):
                target_line_dist = self.distance_line_point(self.points[i-1], self.points[i], target_point)
                if float(target_line_dist) <= float(min_dist):
                    p12_dist = self._distance_point_point(self.points[i-1], self.points[i])
                    p1t_dist = self._distance_point_point(self.points[i-1], target_point)
                    p2t_dist = self._distance_point_point(self.points[i], target_point)
                    if p12_dist > p1t_dist and p12_dist > p2t_dist:
                        entered = True
                        min_dist = target_line_dist
                        idx = i
            if not entered:
                target_point = [x, y]
                sorted_points = sorted(self.points, key=lambda e: self._distance_point_point(e, target_point))
                idx = min(self.points.index(sorted_points[0]), self.points.index(sorted_points[1])) + 1
            self.points.insert(idx, [x, y])

        point = SemSegPointItem(self.last_item_id, self.point_index, self.point_pen, self.point_brush,
                                x, y, 0, 0, self.radius, self.radius)
        point.set_as_shape_point()
        point.set_connection(self.on_point_changed, self.on_edit_finish)
        self.point_index += 1
        self.scene.addItem(point)

        if self.new_shape:
            self.points_dict[self.last_item_id].append(point)
        else:
            self.points_dict[self.last_item_id].insert(idx, point)

        self.segmentedObjects[self.last_item_id].set_points(self.points)
        self.changes.add_item(ChangesItem(self.get_item_details(self.segmentedObjects[self.last_item_id])))
        self.update_save_button(True)
        self._draw_polygon(self.last_item_id)

    def on_instance_id_change(self, value):
        if len(self.scene.selectedItems()) > 0:
            shape = self.scene.selectedItems()[0]
            if shape.__class__ != SemSegPointItem:
                shape.instance_id = value
                self.update_save_button(True)

    # Slots Buttons ----------------------------------------------------------------------------------------------------
    def on_click_clear_frame(self):
        if self.draw_enabled:
            self.on_click_button_draw()
        self.update_save_button(True)

        for key, item in self.segmentedObjects.items():
            self.scene.removeItem(item)
        self.segmentedObjects.clear()

        for key, points in self.points_dict.items():
            for point in points:
                self.scene.removeItem(point)
        self.points_dict.clear()

        # Instances Page
        self.qtCamSegmented.scene().clear()
        self.qtCamInstances.scene().clear()
        self.qtSegmentedTabOriginalImage.scene().clear()
        self.qtSegmentedTabSemanticImage.scene().clear()
        self.qtSegmentedTabInstanceImage.scene().clear()

        self.refresh()

    def on_click_copy_frame(self):
        self.copy_shapes_frame.clear()

        for shape_old in self.segmentedObjects.values():
            shape_info = [self.frame_id,
                          shape_old.item_id,
                          shape_old.get_class(),
                          shape_old.instance_id,
                          [(p.x(), p.y()) for p in shape_old.polygon()]]
            self.copy_shapes_frame.append(shape_info)

    def on_click_paste_frame(self):
        if len(self.segmentedObjects.keys()):
            self.signal_error_signal.emit("Paste frame is for empty frame!!! Frame ID = {} Filter ID = {}"
                                          .format(self.frame_id, self.filter_id), self.filter_id)
            return

        self._load_shapes(self.copy_shapes_frame)
        self.update_save_button(True)

    def on_click_bring_to_front(self):
        if len(self.scene.selectedItems()) == 0:
            return

        shape = self.scene.selectedItems()[0]
        if shape.__class__ == SemSegPointItem:
            return

        shapes = []
        for shape_old in self.segmentedObjects.values():
            if shape_old == shape:
                continue

            shape_info = [self.frame_id,
                          shape_old.item_id,
                          shape_old.get_class(),
                          shape_old.instance_id,
                          [(p.x(), p.y()) for p in shape_old.polygon()]]
            shapes.append(shape_info)

        shape_info = [self.frame_id,
                      shape.item_id,
                      shape.get_class(),
                      shape.instance_id,
                      [(p.x(), p.y()) for p in shape.polygon()]]
        shapes.append(shape_info)

        self._redraw(shapes)

    def on_click_send_to_back(self):
        if len(self.scene.selectedItems()) == 0:
            return

        shape = self.scene.selectedItems()[0]
        if shape.__class__ == SemSegPointItem:
            return

        shapes = []
        shape_info = [self.frame_id,
                      shape.item_id,
                      shape.get_class(),
                      shape.instance_id,
                      [(p.x(), p.y()) for p in shape.polygon()]]
        shapes.append(shape_info)

        for shape_old in self.segmentedObjects.values():
            if shape_old == shape:
                continue

            shape_info = [self.frame_id,
                          shape_old.item_id,
                          shape_old.get_class(),
                          shape_old.instance_id,
                          [(p.x(), p.y()) for p in shape_old.polygon()]]
            shapes.append(shape_info)

        self._redraw(shapes)

    def on_save_finished(self, sem_seg_img, inst_img):
        self._save_data_descriptor(sem_seg_img, inst_img)
        color = self.object_classes.colormap()
        overlay_sem_seg_img = self._process_image(sem_seg_img, color)
        overlay_inst_img = self._process_image(inst_img, color)
        self._load_processed_image_in_view(overlay_sem_seg_img, self.qtCamSegmented)
        self._load_processed_image_in_view(overlay_inst_img, self.qtCamInstances)
        sem_seg_img = self._process_layer_image(sem_seg_img, color)
        inst_img = self._process_layer_image(inst_img, color)
        self._load_processed_image_in_view(sem_seg_img, self.qtSegmentedTabSemanticImage)
        self._load_processed_image_in_view(inst_img, self.qtSegmentedTabInstanceImage)
        QApplication.processEvents()

    # Inherited Slots --------------------------------------------------------------------------------------------------
    def on_main_splitter_moved(self):
        if self.original_image is not None:
            self.scene.setSceneRect(0, 0, self.original_image.shape[1], self.original_image.shape[0])

        self.qtGraphicsView.fitInView(self.qtGraphicsView.scene().sceneRect(), Qt.KeepAspectRatio)
        self.qtCamSegmented.fitInView(self.qtCamSegmented.scene().sceneRect(), Qt.KeepAspectRatio)
        self.qtCamInstances.fitInView(self.qtCamInstances.scene().sceneRect(), Qt.KeepAspectRatio)

        self.qtSegmentedTabOriginalImage.fitInView(self.qtSegmentedTabOriginalImage.scene().sceneRect(),
                                                   Qt.KeepAspectRatio)
        self.qtSegmentedTabSemanticImage.fitInView(self.qtSegmentedTabSemanticImage.scene().sceneRect(),
                                                   Qt.KeepAspectRatio)
        self.qtSegmentedTabInstanceImage.fitInView(self.qtSegmentedTabInstanceImage.scene().sceneRect(),
                                                   Qt.KeepAspectRatio)
        self.zoom = 1

    # Inherited Slots Load Data ----------------------------------------------------------------------------------------
    def on_load_data(self, line_index, ts_sync_timestamp):
        self.clear()

        self.show_warning_text()
        self.show_item_id()
        self.zoom = 1
        self.show_runtime_values(frame_id=line_index-1)
        data, img = super().on_load_data(line_index, ts_sync_timestamp)

        if img is None or data is None:
            return

        ts_stop_img, sampling_time_img, ts_image, cv_img, _ = img
        self.original_image = cv_img
        self._load_annotation_image()
        color = self.object_classes.colormap()

        ts_stop, sampling_time, _, _, _ = data
        self.last_item_id = 0
        self.show_runtime_values(ts_stop=ts_stop, ts_sampling_time=sampling_time, frame_id=self.line_index,
                                 ts_image=ts_image)

        if data[4] is not None:
            if len(data[4]) > 0:
                self._load_shapes(data[4])
        else:
            self.signal_warning_signal.emit("Shapes NONE for line index {}, timestamp sync {}!"
                                            .format(line_index, ts_sync_timestamp), self.filter_id)

        if self.qtWarning.isVisible():
            self._show_points(-1)
            self.on_main_splitter_moved()
            self.update_save_button(False)
            return

        sem_seg = None
        inst = None

        if data[2] is not None:
            if len(data[2]) > 0:
                result = self._process_image(data[2], color)
                sem_seg = self._process_layer_image(data[2], color)
                self._load_processed_image_in_view(result, self.qtCamSegmented, data[2])
        else:
            self.signal_warning_signal.emit("Segmented image NONE for line index {}, timestamp sync {}!"
                                            .format(line_index, ts_sync_timestamp), self.filter_id)

        if data[3] is not None:
            if len(data[3]) > 0:
                result = self._process_image(data[3], color)
                inst = self._process_layer_image(data[3], color)
                self._load_processed_image_in_view(result, self.qtCamInstances)
        else:
            self.signal_warning_signal.emit("Instances image NONE for line index {}, timestamp sync {}!"
                                            .format(line_index, ts_sync_timestamp), self.filter_id)

        self._fill_segmented_layers(sem_seg, inst)
        self._show_points(-1)
        self.on_main_splitter_moved()
        self.update_save_button(False)

    # Inherited Slots Classes ------------------------------------------------------------------------------------------
    def on_highlighted_class(self):
        self.shape_moved_last_position = None
        self.qtInstanceId.setEnabled(False)
        self.qtClasses.clearSelection()
        self.show_item_id()
        self._show_points(index=self.last_item_id, show=True)

        if self.last_item_id in self.segmentedObjects.keys():
            self.segmentedObjects[self.last_item_id].setZValue(0)

        try:
            if self.scene and len(self.scene.selectedItems()) > 0:
                shape = self.scene.selectedItems()[0]
                self.last_item_id = shape.get_item_id()

                if self.last_item_id in self.segmentedObjects.keys():
                    self.segmentedObjects[self.last_item_id].setZValue(1)

                if shape.__class__ != SemSegPointItem:
                    class_id = int(shape.get_class())
                    item = self.get_class_item(class_id)
                    item.setSelected(True)

                    self.qtInstanceId.valueChanged.disconnect()

                    if self._is_countable(shape.class_):
                        self.qtInstanceId.setEnabled(True)
                        self.qtInstanceId.setValue(shape.instance_id)
                    else:
                        self.qtInstanceId.setValue(0)

                    self.qtInstanceId.valueChanged.connect(self.on_instance_id_change)

                self.show_item_id(self.last_item_id)
                self._show_points(index=self.last_item_id, show=True)
            else:
                self._show_points(-1)
        except RuntimeError:
            self.signal_error_signal.emit("RuntimeError - on_highlighted_shape_class()", self.filter_id)

    def on_change_class_item(self):
        self.brush = QBrush(self.draw_class.background(1))
        if len(self.scene.selectedItems()) > 0:
            shape = self.scene.selectedItems()[0]
            if shape.__class__ != SemSegPointItem:
                self.changes.add_item(ChangesItem(self.get_item_details(shape)))
                shape.set_class(int(self.qtClasses.selectedItems()[0].text(2)), self.brush)

                self.qtInstanceId.setEnabled(False)
                if self._is_countable(shape.class_):
                    self.qtInstanceId.setEnabled(True)
                    self.qtInstanceId.setValue(shape.instance_id)
                else:
                    self.qtInstanceId.setValue(0)

                self.update_save_button(True)
                self.scene.update()

    def on_select_class(self, it):
        self.qtClasses.clearSelection()
        it.setSelected(True)
        self.draw_class = it
        self.brush = QBrush(self.draw_class.background(1))
        self.signal_class_selected.emit()

    # Inherited Slots Draw/Delete/Save ---------------------------------------------------------------------------------
    def on_click_button_draw(self):
        if self.draw_enabled:
            self.show_number_of_items(len(self.segmentedObjects.keys()))
            if self.last_item_id in self.segmentedObjects:
                self.segmentedObjects[self.last_item_id].setSelected(False)
            self.draw_enabled = False
            self.signal_activate_draw.emit()
            self.qtButtonDrawShape.setText("Draw Shape")
            self.points.clear()
            self.point_index = 0
        else:
            if self.scene.selectedItems().__len__():
                self.draw_enabled = True
                self.signal_activate_draw.emit()
                self.qtButtonDrawShape.setText("Stop Drawing Shape")
                self.new_shape = False
                self.points = self.segmentedObjects[self.last_item_id].get_coord_points()
                return

            self.new_shape = True
            self.scene.clearSelection()
            self._show_points(-1)
            if self.draw_class is None:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Warning")
                msg.setText("Select a class!")
                msg.exec()
            else:
                self.qtInstanceId.setValue(0)
                self.points = []
                if len(self.segmentedObjects.keys()) > 0:
                    self.last_item_id = max(self.segmentedObjects.keys()) + 1
                self.draw_enabled = True
                self.signal_activate_draw.emit()
                self.qtButtonDrawShape.setText("Stop Drawing Shape")
                class_shape = int(self.draw_class.text(2))
                self._draw_shape(class_shape)

    def on_click_button_delete(self):
        if len(self.scene.selectedItems()) > 0:
            shape = self.scene.selectedItems()[0]
            if shape.__class__ == SemSegPointItem:
                self._delete_point(shape)
                return
            self._delete_shape(shape)
            self.show_number_of_items(len(self.segmentedObjects.keys()))

    def on_click_button_save(self):
        base_path = self.parser.get_parser_base_path(self.filter_id)
        self._save_frame_based_data(base_path)
        self._save_image()
        self.update_save_button(False)
        if self.draw_enabled:
            self.on_click_button_draw()
        self.on_main_splitter_moved()

    # Inherited Slots Copy/Cut/Paste/Undo ------------------------------------------------------------------------------
    def on_copy(self):
        if len(self.scene.selectedItems()) > 0:
            shape = self.scene.selectedItems()[0]
            if shape.__class__ != SemSegPointItem:
                self.clipboard = ["copy", self.get_item_details(shape)]

    def on_cut(self):
        if len(self.scene.selectedItems()) > 0:
            shape = self.scene.selectedItems()[0]
            if shape.__class__ != SemSegPointItem:
                self.clipboard = ["cut", self.get_item_details(shape)]
                self.on_click_button_delete()
                self.update_save_button(True)

    def on_paste(self):
        if len(self.scene.selectedItems()) != 0:
            for shape in self.scene.selectedItems():
                if shape.__class__ != SemSegPointItem:
                    shape.setSelected(False)

        if self.clipboard is None:
            return

        shape = self.clipboard[1].replace('\n', '').split(',')

        timestamp_stop = shape[0]
        item_id = int(shape[1])
        item_c = int(shape[2])
        item_i = int(shape[3])
        item_p = shape[4]
        points = []

        try:
            points_str = item_p[2:-2]
            points_str = points_str.split("][")
            points = list()

            for point in points_str:
                point = point.split(" ")
                points.append([float(point[0]), float(point[1])])
        except RuntimeError:
            self.signal_error_signal.emit("RuntimeError - on_paste()", self.filter_id)

        if self.clipboard[0] == "copy" or str(timestamp_stop) != str(self.frame_id):
            if len(self.segmentedObjects) > 0:
                self.last_item_id = max(self.segmentedObjects.keys()) + 1
            else:
                self.last_item_id = 1
        else:
            self.last_item_id = item_id

        shape = self._draw_shape(class_index=item_c, shape_instance=item_i, points=points)
        self.changes.add_item(ChangesItem(self.get_item_details(shape), was_new=True))
        self.show_number_of_items(self.segmentedObjects.keys().__len__())
        self.update_save_button(True)

    def on_undo(self, item):
        details = str(item.item)
        shape = details.replace('\n', '').split(',')

        item_id = int(shape[1])
        item_c = int(shape[2])
        item_i = int(shape[3])
        item_p = shape[4]
        points = []

        try:
            points_str = item_p[2:-2]
            points_str = points_str.split("][")
            points = list()

            for point in points_str:
                point = point.split(" ")
                points.append([float(point[0]), float(point[1])])
        except RuntimeError:
            self.signal_error_signal.emit("RuntimeError - on_undo for item id {}".format({item_id}), self.filter_id)
            return

        if item.was_removed:
            self.last_item_id = item_id
            self._draw_shape(class_index=item_c, shape_instance=item_i, points=points, position=int(shape[5]))

        if item_id in self.segmentedObjects.keys():
            self.last_item_id = item_id
            shape = self.segmentedObjects[item_id]
            shape.setSelected(True)

            if item.was_new:
                self._delete_shape(shape)
            else:
                self._update_polygon(item_c, points)

        self.scene.update()
        self.scene.clearSelection()
        self.show_number_of_items(self.segmentedObjects.keys().__len__())
