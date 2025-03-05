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
from tools.Annotation.items.Roi2DItem import Roi2DItem
from tools.Annotation.items.DojoGraphicScene import DojoGraphicScene

from toolkit.env.object_classes import ObjectClasses

from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import os
import sys

sys.path.insert(1, '../')
current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")


class Roi2DWindow(FilterWindowInterface):
    # Signals ----------------------------------------------------------------------------------------------------------
    signal_class_selected = pyqtSignal()

    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id, input_sources=None, road=False):
        super().__init__(filter_id, input_sources=input_sources)
        uic.loadUi(os.path.join(ui_directory, 'Roi2DWindow.ui'), self)
        self.setWindowTitle("Roi2D Viewer")

        # Widget - children
        self.find_children()

        # Widget - Buttons
        self.qtButtonDrawROI = self.findChild(QPushButton, 'qtButtonDrawROI')
        self.qtButtonDeleteROI = self.findChild(QPushButton, 'qtButtonDeleteROI')

        # Widget
        self.qtRoiID = self.findChild(QSpinBox, 'qtRoiID')

        # Members
        self.object_classes = ObjectClasses(self.object_classes_path)
        self.scene = DojoGraphicScene()
        self.pen_size = 5
        self.temp_roi = None
        self.draw_class = None
        self.frame_based_header = "frame_id,roi_id,cls,x,y,width,height\n"
        self.road = road

        # Initialization
        self.init_connections()
        self.init_tree_widget()
        self.init_combo_overlay()
        self.init_window()
        self.init_splitter()

    # Inherited Initialization Methods ---------------------------------------------------------------------------------
    def init_window(self):
        self.qtGraphicsView.setScene(self.scene)

        self.image_timestamps = self.parser.get_all_timestamps(filter_id=self.filter_id_image)
        self.synced_timestamps = self.parser.get_records_synced(filter_ids=[self.filter_id_image, self.filter_id])

        self.qtWarning.setText("ATTENTION: Class index out of bounds. Please check your classes file!")
        self.qtWarning.setStyleSheet("QLabel {background-color:red}")

        self.show_warning_text(False)
        self.update_save_button(False)

    def init_connections(self):
        self.qtRoiID.valueChanged.connect(self.on_change_roi_id)
        self.qtButtonDrawROI.clicked.connect(self.on_click_button_draw)
        self.qtButtonDeleteROI.clicked.connect(self.on_click_button_delete)
        self.qtButtonSave.clicked.connect(self.on_click_button_save)
        self.qtClasses.itemClicked.connect(self.on_select_class)
        self.qtComboOverlay.currentTextChanged.connect(self.on_change_text_combo)
        self.scene.signal_draw_new_roi.connect(self.on_draw_new_roi)
        self.scene.signal_draw_temp_roi.connect(self.on_draw_temp_roi)
        self.scene.selectionChanged.connect(self.on_highlighted_class)
        self.signal_class_selected.connect(self.on_change_class_item)

    # Inherited Methods ------------------------------------------------------------------------------------------------
    def clear(self):
        if self.draw_enabled:
            self.on_click_button_draw()

        self.update_save_button(False)

        if len(self.items) != 0:
            if self.qtButtonSave.isEnabled():
                self.on_click_button_save()

        self.items.clear()
        self.scene.clear()

    def get_item_details(self, box):
        return "{},{},{},{},{},{},{}\n".format(self.frame_id,
                                               int(box.get_item_id()),
                                               int(box.get_class()),
                                               int(box.sceneBoundingRect().x()),
                                               int(box.sceneBoundingRect().y()),
                                               int(box.sceneBoundingRect().width()),
                                               int(box.sceneBoundingRect().height()))

    # Private Methods --------------------------------------------------------------------------------------------------
    def _new_item_id_is_valid(self):
        new_id = self.qtRoiID.value()
        for roi in self.items:
            if roi.get_item_id() == new_id:
                return False
        return True

    # Private Methods - Classes ----------------------------------------------------------------------------------------
    def _get_class_color(self, class_idx):
        item = self.get_class_item(class_idx)

        if item is not None:
            return item.background(1)
        else:
            return QBrush(Qt.white)

    def _check_invalid_classes(self):
        if self.qtWarning.isVisible():
            for roi in self.items:
                if len(self.object_classes.object_classes) < int(roi.get_class()):
                    self.show_warning_text(True)
                    self.signal_error_signal.emit("Invalid object class file!", self.filter_id)

        self.show_warning_text(False)

    # Private Methods - ROIs -------------------------------------------------------------------------------------------
    def _load_rois(self, rois):
        self.show_number_of_items(len(rois))
        for roi in rois:
            self.frame_id = int(roi[0])
            self.last_item_id = int(roi[1])
            self.qtFrameID.setText(str(self.frame_id))
            pen_roi = QPen(self._get_class_color(int(roi[2])), self.pen_size)
            qtRoi = self._draw_roi(roi[3], roi[4], roi[5], roi[6], pen_roi, roi[2])

            if self.road:
                qtRoi.actions = str(roi[7])
                qtRoi.locs = str(roi[8])
        self.on_main_splitter_moved()

    def _draw_roi(self, x, y, w, h, pen, class_id, temp=False, roi_id=-1):
        if self.draw_enabled:
            self.update_save_button(True)

        if self.temp_roi is not None:
            self.scene.removeItem(self.temp_roi)

        if temp:
            self.temp_roi = Roi2DItem(self.last_item_id, class_id, pen, x, y, w, h)
            self.scene.addItem(self.temp_roi)
        else:
            roi_id_ = roi_id
            if roi_id < 0:
                roi_id_ = self.last_item_id

            rect_item = Roi2DItem(roi_id_, class_id, pen, x, y, w, h)
            rect_item.set_connection(self.on_item_changed, self.on_edit_finish)
            self.scene.addItem(rect_item)
            self.items.append(rect_item)
            self.show_number_of_items(len(self.items))
            return rect_item
        return None
        QApplication.processEvents()

    def _delete_roi(self, box):
        self.update_save_button(True)
        self.items.remove(box)
        self.scene.removeItem(box)
        self.show_number_of_items(len(self.items))

    # Slots ROIs -------------------------------------------------------------------------------------------------------
    def on_draw_temp_roi(self, x, y, w, h):
        pen_roi = QPen(self.draw_class.background(1), self.pen_size)
        self._draw_roi(x, y, w, h, pen_roi, (self.qtClasses.currentIndex().row() + 1), True)

    def on_draw_new_roi(self, x, y, w, h):
        self.last_item_id += 1
        pen_roi = QPen(self.draw_class.background(1), self.pen_size)
        class_roi = self.qtClasses.currentIndex().row()
        new_roi = self._draw_roi(x, y, w, h, pen_roi, class_roi)
        self.changes.add_item(ChangesItem(self.get_item_details(new_roi), was_new=True))
        self.on_click_button_draw()

    def on_change_roi_id(self):
        if len(self.scene.selectedItems()) > 0:
            if not self._new_item_id_is_valid():
                self.signal_error_signal.emit("Roi ID {} already exists!"
                                              .format(self.qtRoiID.value()), self.filter_id)
                return
            box = self.scene.selectedItems()[0]
            old_id = box.get_item_id()
            box.set_item_id(self.qtRoiID.value())
            self.show_item_id(self.qtRoiID.value())
            self.changes.add_item(ChangesItem(self.get_item_details(box), old_id=old_id))
        else:
            self.qtRoiID.setValue(-1)

    # Inherited Slots Load Data ----------------------------------------------------------------------------------------
    def on_load_data(self, line_index, ts_sync_timestamp):
        self.show_warning_text(False)
        self.show_runtime_values(frame_id=line_index - 1)
        data, img = super().on_load_data(line_index, ts_sync_timestamp)
        self.clear()

        if img is None or data is None:
            return

        if self.road:
            ts_stop_rois, sampling_time_rois, ego_id, rois = data
            self.signal_info_signal.emit("EGO ID {}!".format(ego_id), self.filter_id)
        else:
            ts_stop_rois, sampling_time_rois, rois = data
        ts_stop, sampling_time, ts_image, cv_image, _ = img
        self.load_image(cv_image)

        self.show_runtime_values(ts_stop=ts_stop_rois, ts_sampling_time=sampling_time_rois,
                                 ts_image=ts_image, frame_id=self.frame_id)

        if rois is not None:
            self.last_item_id = 0
            self._load_rois(rois)
        else:
            self.signal_warning_signal.emit("Rois NONE for line index {}, timestamp sync {}!"
                                            .format(line_index, ts_sync_timestamp), self.filter_id)

    # Inherited Slots Classes ------------------------------------------------------------------------------------------
    def on_highlighted_class(self):
        self.qtRoiID.valueChanged.disconnect()
        self.qtClasses.clearSelection()
        self.show_item_id()
        self.qtRoiID.setValue(-1)
        self.qtRoiID.setEnabled(False)

        if self.scene and len(self.scene.selectedItems()) > 0:
            box = self.scene.selectedItems()[0]
            class_id = int(box.get_class())
            self.qtRoiID.setEnabled(True)
            self.qtRoiID.setValue(box.get_item_id())
            self.show_item_id(box.get_item_id())

            try:
                item = self.get_class_item(class_id)
                item.setSelected(True)
            except RuntimeError:
                self.signal_error_signal.emit("RuntimeError - on_highlighted_shape_class()", self.filter_id)

        self.qtRoiID.valueChanged.connect(self.on_change_roi_id)

    def on_change_class_item(self):
        if len(self.scene.selectedItems()) > 0:
            box = self.scene.selectedItems()[0]
            self.changes.add_item(ChangesItem(self.get_item_details(box)))
            box.set_class((self.qtClasses.currentIndex().row()))
            box.set_pen(QPen(self.draw_class.background(1), self.pen_size))
            self.scene.update()
            self._check_invalid_classes()
            self.update_save_button(True)

    def on_select_class(self, it):
        self.draw_class = it
        self.signal_class_selected.emit()

    # Inherited Slots Draw/Delete/Save ---------------------------------------------------------------------------------
    def on_click_button_draw(self):
        if self.draw_enabled:
            self.draw_enabled = False
            self.scene.on_active_draw()
            self.qtButtonDrawROI.setText("Draw ROI")
        else:
            if self.draw_class is None:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Warning")
                msg.setText("Select a class!")
                msg.exec()
            else:
                self.draw_enabled = True
                self.scene.on_active_draw()
                self.qtButtonDrawROI.setText("Stop Drawing ROI")

    def on_click_button_delete(self):
        if len(self.scene.selectedItems()) > 0:
            box = self.scene.selectedItems()[0]
            self.changes.add_item(ChangesItem(self.get_item_details(box), was_removed=True))
            self._delete_roi(box)

    # Inherited Slots Copy/Cut/Paste/Undo ------------------------------------------------------------------------------
    def on_copy(self):
        if len(self.scene.selectedItems()) > 0:
            box = self.scene.selectedItems()[0]
            self.clipboard = ["copy", self.get_item_details(box)]
            self.update_save_button(True)

    def on_cut(self):
        if len(self.scene.selectedItems()) > 0:
            box = self.scene.selectedItems()[0]
            self.clipboard = ["cut", self.get_item_details(box)]
            self.on_click_button_delete()
            self.update_save_button(True)

    def on_paste(self):
        if self.clipboard is None:
            return

        roi = self.clipboard[1].replace('\n', '').split(',')

        frame_id = int(roi[0])
        item_id = int(roi[1])
        item_c = int(roi[2])
        item_x = int(roi[3])
        item_y = int(roi[4])
        item_w = int(roi[5])
        item_h = int(roi[6])

        if self.clipboard[0] == "copy" or frame_id != self.frame_id:
            item_id = -1
            self.last_item_id += 1

        pen_roi = QPen(self.get_class_item(item_c).background(1), self.pen_size)
        box = self._draw_roi(x=item_x, y=item_y, w=item_w, h=item_h, pen=pen_roi, class_id=item_c, roi_id=item_id)
        self.changes.add_item(ChangesItem(self.get_item_details(box), was_new=True))
        self.clipboard = None
        self.show_number_of_items(self.items.__len__())
        self.update_save_button(True)

    def on_undo(self, item):
        if self.clipboard is None:
            self.clipboard = None
        details = str(item.item)
        roi = details.replace('\n', '').split(',')
        frame_id = int(roi[0])

        if self.frame_id != frame_id:
            return

        item_id = int(roi[1])
        item_c = int(roi[2])
        item_x = int(roi[3])
        item_y = int(roi[4])
        item_w = int(roi[5])
        item_h = int(roi[6])
        color = self._get_class_color(item_c)
        pen_roi = QPen(color, self.pen_size)

        if item.was_removed:
            self._draw_roi(x=item_x, y=item_y, w=item_w, h=item_h, pen=pen_roi, class_id=item_c, roi_id=item_id)

        box = self.find_item_by_id(item_id)
        if box is not None:
            self._delete_roi(box)
            if item.old_id != -1:
                self._draw_roi(x=item_x, y=item_y, w=item_w, h=item_h, pen=pen_roi, class_id=item_c, roi_id=item.old_id)
            else:
                if not item.was_new:
                    self._draw_roi(x=item_x, y=item_y, w=item_w, h=item_h, pen=pen_roi, class_id=item_c, roi_id=item_id)

        self.scene.update()
        self.scene.clearSelection()
        self.show_number_of_items(self.items.__len__())
