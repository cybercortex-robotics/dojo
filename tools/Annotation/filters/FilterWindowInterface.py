"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from data.CyC_db_interface import CyC_DatabaseParser
from global_config import *

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QColor, QBrush
from PyQt5.QtWidgets import *
from PyQt5 import QtGui

import shutil


class ChangesItem:
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, item, was_new=False, was_removed=False, old_id=-1):
        self.item = item
        self.was_new = was_new
        self.was_removed = was_removed
        self.old_id = old_id

    # Methods ----------------------------------------------------------------------------------------------------------
    def equals(self, item):
        if self.item != item.item:
            return False
        return False


class ChangesStack(QObject):
    # Signals ----------------------------------------------------------------------------------------------------------
    signal_revert_change = pyqtSignal(ChangesItem)

    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self):
        QObject.__init__(self)
        self.limit = 100
        self.stack = []

    # Methods ----------------------------------------------------------------------------------------------------------
    def exists(self, item):
        for elem in self.stack:
            if elem.equals(item):
                return True
        return False

    def add_item(self, item):
        if len(self.stack) == self.limit:
            self.stack.pop(0)

        if item.was_new or item.was_removed:
            self.stack.append(item)
        else:
            if not self.exists(item):
                self.stack.append(item)

    def undo(self):
        if len(self.stack) > 0:
            last_change = self.stack.pop()
            self.signal_revert_change.emit(last_change)


class FilterWindowInterface(QWidget):
    # Signals ----------------------------------------------------------------------------------------------------------
    signal_info_signal = pyqtSignal(str, int)
    signal_warning_signal = pyqtSignal(str, int)
    signal_error_signal = pyqtSignal(str, int)
    signal_disable_play_menu = pyqtSignal(bool)
    signal_activate_draw = pyqtSignal()

    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id, input_sources=None, is_static=False):
        super().__init__()

        # Widgets
        self.qtCalibrationTextEdit = None
        self.qtTsSamplingTime = None
        self.qtGraphicsView = None
        self.qtPlotWidget = None
        self.qtComboOverlay = None
        self.qtButtonSave = None
        self.qtTsImage = None
        self.qtTsStop = None
        self.qtFrameID = None
        self.splitter = None
        self.qtNoItems = None
        self.qtItemID = None
        self.qtWarning = None
        self.qtClasses = None

        # Members
        self.input_sources = input_sources
        self.filter_id = filter_id
        self.is_static = is_static
        self.frame_id = 0
        self.line_index = -1
        self.frame_based_header = None
        self.is_processing = False
        self.image_size = None
        self.scene = None
        self.parser = CyC_DatabaseParser.get_instance(cfg.DB.BASE_PATH)
        self.timestamps = self.parser.get_all_timestamps(filter_id=self.filter_id)
        self.last_item_id = -1
        self.filter_id_image = 1
        self.image_timestamps = None
        self.synced_timestamps = None
        self.items = []
        self.edit_finish = True
        self.draw_enabled = False
        self.clipboard = None
        self.changes = ChangesStack()

        # Calibration and object classes files
        base_path = self.parser.get_parser_base_path(self.filter_id)

        self.calibration_path = os.path.join(base_path, 'calibration.cal')
        self.object_classes_path = os.path.join(base_path, 'object_classes.conf')

        if not os.path.exists(self.calibration_path):
            self.calibration_path = None

        if not os.path.exists(self.object_classes_path):
            self.object_classes_path = None

        # Signals
        self.changes.signal_revert_change.connect(self.on_undo)

    # Inherited Events -------------------------------------------------------------------------------------------------
    def resizeEvent(self, event):
        self.on_main_splitter_moved()

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Insert:
            self.on_click_button_draw()

        if key == Qt.Key_Delete:
            self.on_click_button_delete()

        ctrl = False
        if event.modifiers() and Qt.ControlModifier:
            ctrl = True

        if ctrl and key == Qt.Key_S:
            self.on_click_button_save()

        if ctrl and key == Qt.Key_Z:
            self.changes.undo()

        if ctrl and key == Qt.Key_X:
            self.on_cut()

        if ctrl and key == Qt.Key_C:
            self.on_copy()

        if ctrl and key == Qt.Key_V:
            self.on_paste()

        super().keyPressEvent(event)

    # Initialization Methods -------------------------------------------------------------------------------------------

    def find_children(self):
        self.qtGraphicsView = self.findChild(QGraphicsView, 'qtGraphicsView')
        self.qtButtonSave = self.findChild(QPushButton, 'qtButtonSave')
        self.qtFrameID = self.findChild(QLabel, 'qtFrameID')
        self.qtItemID = self.findChild(QLabel, 'qtItemID')
        self.qtComboOverlay = self.findChild(QComboBox, 'qtComboOverlay')
        self.qtTsSamplingTime = self.findChild(QLineEdit, 'qtTsSamplingTime')
        self.qtTsImage = self.findChild(QLineEdit, 'qtTsImage')
        self.qtTsStop = self.findChild(QLineEdit, 'qtTsStop')
        self.qtNoItems = self.findChild(QLabel, 'qtNoItems')
        self.splitter = self.findChild(QSplitter, 'splitter')
        self.qtWarning = self.findChild(QLabel, 'qtWarning')
        self.qtClasses = self.findChild(QTreeWidget, 'qtClasses')
        # self.qtCalibrationTextEdit = self.findChild(QTextEdit, 'qtCalibrationTextEdit')
        # self.qtPlotWidget = self.findChild(QWidget, 'qtPlotWidget')

    def init_combo_overlay(self):
        if self.qtComboOverlay is None:
            return

        self.qtComboOverlay.clear()
        if self.input_sources:
            image_stream = self.input_sources[0]
            data_type, stream_id = image_stream.split("-")
            self.filter_id_image = stream_id

            stream = 'datastream_{}'.format(self.filter_id_image)
            self.qtComboOverlay.addItem(stream)
            self.qtComboOverlay.setCurrentText(stream)

    def init_window(self):
        pass

    def init_tree_widget(self):
        self.qtClasses.setHeaderHidden(True)
        self.qtClasses.setColumnCount(3)
        self.qtClasses.setColumnWidth(0, 150)
        self.qtClasses.setColumnWidth(1, 25)
        self.qtClasses.clear()
        colormap = self.object_classes.colormap()

        for obj_class in self.object_classes.object_classes:
            class_color_blue = float(colormap[obj_class[0]][0]) / 255.
            class_color_green = float(colormap[obj_class[0]][1]) / 255.
            class_color_red = float(colormap[obj_class[0]][2]) / 255.

            qColor = QColor(0, 0, 0)
            qColor.setRedF(class_color_red)
            qColor.setGreenF(class_color_green)
            qColor.setBlueF(class_color_blue)

            parent = QTreeWidgetItem(self.qtClasses)
            parent.setText(0, str(obj_class[1]))
            parent.setBackground(1, QBrush(qColor))
            parent.setText(2, str(obj_class[0]))

    def init_connections(self):
        pass

    def init_splitter(self):
        if self.splitter is None:
            return

        self.splitter.splitterMoved.connect(self.on_main_splitter_moved)
        self.splitter.setSizes([500, 100])

    # STATIC Methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def default_info_warning_error_slot(message, code):
        print("CODE {}: ".format(message, code))

    # Methods ----------------------------------------------------------------------------------------------------------
    def clear(self):
        pass

    def refresh(self):
        pass

    def get_item_details(self, item):
        return ""

    def get_class_item(self, class_idx):
        if self.object_classes.get_class_max_index() < class_idx:
            self.show_warning_text(True)
            self.signal_error_signal.emit("Invalid object class file!", self.filter_id)
            return None

        class_name = self.object_classes.get_name_by_index(class_idx)
        items = self.qtClasses.findItems(class_name, Qt.MatchContains, 0)

        for item in items:
            if item.text(0) == class_name:
                return item

        self.signal_error_signal.emit("Class item id {} not found!".format(class_idx), self.filter_id)
        return None

    def show_warning_text(self, value=False):
        self.qtWarning.setVisible(value)

    def update_save_button(self, modified):
        if self.qtButtonSave is None:
            return

        if modified:
            self.qtButtonSave.setStyleSheet("background-color: green")
            self.qtButtonSave.setToolTip('')
        else:
            self.qtButtonSave.setStyleSheet("background-color: gray")
            self.qtButtonSave.setToolTip('No changes!')
        self.qtButtonSave.setEnabled(modified)

    def show_calibration_file(self):
        if self.calibration_path is not None:
            with open(self.calibration_path, "r") as calibration_file:
                content = calibration_file.read()
                self.qtCalibrationTextEdit.setTextColor(Qt.black)
                self.qtCalibrationTextEdit.setText(content)

    def show_number_of_items(self, value):
        if self.qtNoItems is None:
            return

        self.qtNoItems.setText(str(value))

    def show_item_id(self, value=-1):
        if self.qtItemID is None or self.scene is None:
            return

        if value > 0:
            self.qtItemID.setText(str(value))
        else:
            self.qtItemID.setText("No item selected!")

    def save_calibration_file(self):
        if self.calibration_path is None:
            base_path = self.parser.get_parser_base_path(self.filter_id)
            self.calibration_path = os.path.join(base_path, 'calibration.cal')

        with open(self.calibration_path, 'w') as calibration_file:
            calibration_file.write(str(self.qtCalibrationTextEdit.toPlainText()))

    def data_is_valid(self, data):
        if data is None:
            self.signal_error_signal.emit("Data NONE!", self.filter_id)
            return False
        for d in range(len(data) - 1):
            if d is None:
                return False
        return True

    def show_runtime_values(self, frame_id=0, ts_stop="", ts_sampling_time="", ts_image=""):
        if self.qtTsSamplingTime is not None:
            self.qtTsSamplingTime.setText(str(ts_sampling_time))

        if self.qtTsStop is not None:
            self.qtTsStop.setText(str(ts_stop))

        if self.qtTsImage is not None:
            self.qtTsImage.setText(str(ts_image))

        if self.qtFrameID is not None:
            self.qtFrameID.setText(str(frame_id))
            self.frame_id = frame_id

    def load_image(self, cv_image):
        if self.scene is None:
            return

        if cv_image is None:
            self.signal_error_signal.emit("Image NONE!", self.filter_id)
            return

        if len(cv_image) < 0:
            self.signal_error_signal.emit("Empty image data!", self.filter_id)
            return

        self.clear()
        if self.image_size is None:
            self.image_size = (cv_image.shape[0], cv_image.shape[1])
        image = QtGui.QImage(cv_image, cv_image.shape[1], cv_image.shape[0], cv_image.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        self.scene.addPixmap(QPixmap(image))
        self.on_main_splitter_moved()

    def load_image_in_view(self, cv_image, view):
        if cv_image is None:
            self.signal_error_signal.emit("Data NONE!", self.filter_id)
            return

        image = QtGui.QImage(cv_image, cv_image.shape[1], cv_image.shape[0], cv_image.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        canvas = QPixmap(cv_image.shape[1], cv_image.shape[0])
        canvas.fill(QColor(0, 0, 0))
        view.scene().addPixmap(canvas)
        image = QPixmap(image)
        view.scene().addPixmap(image)
        view.fitInView(view.scene().sceneRect(), Qt.KeepAspectRatio)

    def find_item_by_id(self, _id):
        for item in self.items:
            if item.get_item_id() == _id:
                return item
        self.signal_warning_signal.emit("Item id {} not found!".format(_id), self.filter_id)
        return None

    # Slots Load Data --------------------------------------------------------------------------------------------------
    def on_load_all_data(self):
        pass

    def on_load_data(self, line_idx, ts_sync_timestamp):
        if self.qtFrameID is None:
            return

        if self.synced_timestamps is None:
            self.signal_error_signal.emit("Synced timestamps NONE for line index {}, timestamp sync {}!"
                                          .format(line_idx, ts_sync_timestamp), self.filter_id)
            return

        if line_idx - 1 >= len(self.synced_timestamps):
            self.signal_error_signal.emit("Line index out of bounds for line index {}, timestamp sync {}!"
                                          .format(line_idx, ts_sync_timestamp), self.filter_id)
            return
        self.line_index = line_idx - 1

        data = self.parser.get_data_at_timestamp(filter_id=self.filter_id,
                                                 ts_stop=self.synced_timestamps[self.line_index][1])

        if not self.data_is_valid(data):
            self.signal_error_signal.emit("Invalid data for line index {}, timestamp sync {}!"
                                          .format(self.line_index, ts_sync_timestamp), self.filter_id)
            return data, None

        img = self.parser.get_data_at_timestamp(filter_id=self.filter_id_image,
                                                ts_stop=self.synced_timestamps[self.line_index][0])
        if not self.data_is_valid(img):
            self.signal_error_signal.emit("Invalid data image filter id {}, line index {}, timestamp sync {}!"
                                          .format(self.filter_id_image, self.line_index, ts_sync_timestamp),
                                          self.filter_id)
            return data, img

        if self.image_size is not None:
            if int(img[3].shape[0]) != int(self.image_size[0]) or int(img[3].shape[1]) != int(self.image_size[1]):
                self.signal_error_signal.emit(
                    "Image does not have the valid size, image filter id {}, line index {}, timestamp sync {}!".format
                    (self.filter_id_image, self.line_index, ts_sync_timestamp), self.filter_id)
                return data, None

        self.last_item_id = 0
        self.frame_id = self.parser.get_frame_id_by_timestamp(self.filter_id, data[0])
        self.show_runtime_values(ts_stop=data[0], ts_sampling_time=data[1], frame_id=self.frame_id)

        return data, img

    # Slots Classes ----------------------------------------------------------------------------------------------------
    def on_highlighted_class(self):
        pass

    def on_change_class_item(self):
        pass

    def on_select_class(self, it):
        pass

    # Slots Item -------------------------------------------------------------------------------------------------------
    def on_item_changed(self, x, y, w, h):
        self.update_save_button(True)
        if len(self.scene.selectedItems()) > 0:
            item = self.scene.selectedItems()[0]
            if self.edit_finish:
                self.changes.add_item(ChangesItem(self.get_item_details(item)))
                self.edit_finish = False
        self.update_save_button(True)

    def on_edit_finish(self):
        self.edit_finish = True

    # Slots Widget -----------------------------------------------------------------------------------------------------
    def on_change_text_combo(self):
        self.filter_id_image = int(self.qtComboOverlay.currentText().split("_")[1])
        self.image_timestamps = self.parser.get_all_timestamps(filter_id=self.filter_id_image)
        self.synced_timestamps = self.parser.get_records_synced(filter_ids=[self.filter_id_image, self.filter_id])

        base_path = self.parser.get_parser_base_path(self.filter_id_image)
        self.calibration_path = os.path.join(base_path, 'calibration.cal')

        if not os.path.exists(self.calibration_path):
            self.calibration_path = None

        if self.line_index != -1:
            self.on_load_data(self.line_index, self.ts_sync)

    def on_main_splitter_moved(self):
        if self.qtGraphicsView is None or self.scene is None:
            return

        self.qtGraphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    # Slots Draw/Delete/Save -------------------------------------------------------------------------------------------
    def on_click_button_save(self):
        if self.frame_based_header is None:
            self.signal_error_signal.emit("Frame based data descriptor header NONE!", self.filter_id)
            return

        base_path = self.parser.get_parser_base_path(self.filter_id)
        file_path = os.path.join(base_path, 'framebased_data_descriptor.csv')
        file_copy_path = os.path.join(base_path, 'framebased_data_descriptor_copy.csv')
        shutil.copyfile(file_path, file_copy_path)
        wrote = False

        if not os.path.exists(file_path):
            self.signal_warning_signal.emit("File {} does not exist!".format(file_path), self.filter_id)

        with open(file_path, "r") as f_in:
            lines = f_in.readlines()

        with open(file_path, "w") as f_out:
            for line in lines:
                if line == self.frame_based_header:
                    f_out.write(line)
                else:
                    try:
                        line_frame_id = int(line.split(",")[0])
                    except RuntimeError:
                        self.signal_error_signal.emit("Frame based data descriptor invalid header!", self.filter_id)
                        return
                    if line_frame_id == self.frame_id:
                        if wrote is False:
                            wrote = True
                            try:
                                self.items = sorted(self.items, key=lambda _item: _item.item_id)
                            except RuntimeError:
                                self.signal_error_signal.emit("Sorting failed!", self.filter_id)
                            for item in self.items:
                                f_out.write(self.get_item_details(item))
                    else:
                        f_out.write(line)
            if wrote is False:
                for item in self.items:
                    f_out.write(self.get_item_details(item))

        os.remove(file_copy_path)
        if self.draw_enabled:
            self.on_click_button_draw()

        self.update_save_button(False)

    def on_click_button_delete(self):
        pass

    def on_click_button_draw(self):
        pass

    # Slots Copy/Cut/Paste/Undo ----------------------------------------------------------------------------------------
    def on_copy(self):
        pass

    def on_cut(self):
        pass

    def on_paste(self):
        pass

    def on_undo(self, item):
        pass
