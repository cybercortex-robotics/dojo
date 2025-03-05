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
from tools.Annotation.items.PointsTrackingItem import PointsTrackingItem, PointsTrackingLineItem
from tools.Annotation.items.DojoGraphicScene import DojoGraphicScene

from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import shutil
import cv2
import os
import sys
import random

sys.path.insert(1, '../')
current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "../ui")


class PointsTrackingWindow(FilterWindowInterface):
    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, filter_id, input_sources=None):
        super().__init__(filter_id)
        uic.loadUi(os.path.join(ui_directory, 'PointsTrackingWindow.ui'), self)
        self.setWindowTitle("Points Tracking Viewer")

        # Widget
        self.find_children()

        self.qtButtonDrawPoints = self.findChild(QPushButton, 'qtButtonDrawPoints')
        self.qtButtonDeletePoint = self.findChild(QPushButton, 'qtButtonDeletePoint')
        self.qtButtonCopyPoints = self.findChild(QPushButton, 'qtButtonCopyPoints')

        self.qtPointID = self.findChild(QSpinBox, 'qtPointID')
        self.qtXCursor = self.findChild(QLabel, 'qtXCursor')
        self.qtYCursor = self.findChild(QLabel, 'qtYCursor')

        # Members
        self.frame_based_header = "frame_id,x,y,id,score\n"
        self.input_sources = input_sources
        self.scene = DojoGraphicScene()
        self.point_items = dict()
        self.color_dict = dict()
        self.lines = dict()
        self.diameter = 25
        self.line_width = 15
        self.copy_points = None
        self.prev_image = None
        self.prev_points = None
        self.offset_x = 1600
        self.last_frame_id = None

        # Initialization
        self.init_connections()
        self.init_window()
        self.init_splitter()

    # Inherited Initialization Methods ---------------------------------------------------------------------------------
    def init_window(self):
        self.qtGraphicsView.setScene(self.scene)
        self.on_main_splitter_moved()
        self.init_combo_overlay()
        self.image_timestamps = self.parser.get_all_timestamps(filter_id=self.filter_id_image)
        self.synced_timestamps = self.parser.get_records_synced(filter_ids=[self.filter_id_image, self.filter_id])
        self.update_save_button(False)

    def init_connections(self):
        self.scene.signal_draw_new_point.connect(self.on_draw_new_point)
        self.scene.selectionChanged.connect(self.on_display_item_id)
        self.scene.signal_cursor_position.connect(self.on_display_cursor_position)
        self.qtComboOverlay.currentTextChanged.connect(self.on_change_text_combo)
        self.qtButtonSave.clicked.connect(self.on_click_button_save)
        self.qtButtonDrawPoints.clicked.connect(self.on_draw_points)
        self.qtButtonCopyPoints.clicked.connect(self.on_copy_paste_points)
        self.qtButtonDeletePoint.clicked.connect(self.on_click_button_delete)
        self.qtPointID.valueChanged.connect(self.on_change_point_id)

    # Inherited Methods ------------------------------------------------------------------------------------------------
    def clear(self):
        self.point_items.clear()
        self.lines.clear()
        self.scene.clear()

    def on_item_changed(self, x, y, w, h):
        self.update_save_button(True)
        if len(self.scene.selectedItems()) > 0:
            item = self.scene.selectedItems()[0]
            if self.edit_finish:
                self.changes.add_item(ChangesItem(self.get_item_details(item)))
                self.edit_finish = False
                self.redraw_line(item)
        self.update_save_button(True)

    def on_click_button_save(self):
        if self.frame_based_header is None:
            self.signal_error_signal.emit("Frame based data descriptor header is none!", self.filter_id)
            return

        base_path = self.parser.get_parser_base_path(self.filter_id)
        file_path = os.path.join(base_path, 'framebased_data_descriptor.csv')
        file_copy_path = os.path.join(base_path, 'framebased_data_descriptor_copy.csv')
        shutil.copyfile(file_path, file_copy_path)
        wrote_frame_id = False
        wrote_last_frame_id = False

        if not os.path.exists(file_path):
            self.signal_warning_signal.emit("File {} does not exist!".format(file_path), self.filter_id)

        with open(file_path, "r") as f_in:
            lines = f_in.readlines()

        current_points  = ""
        previous_points = ""
        for point_id, points in self.point_items.items():
            for point in points:
                prev = self.offset_x < point.pos().x()
                if prev:
                    previous_points += self.get_item_details(point, prev)
                else:
                    current_points += self.get_item_details(point)

        if self.frame_id == self.last_frame_id:
            previous_points = ""

        with open(file_path, "w") as f_out:
            for line in lines:
                if line == self.frame_based_header:
                    f_out.write(line)
                else:
                    try:
                        line_frame_id = int(line.split(",")[0])
                    except:
                        self.signal_error_signal.emit("Frame based data descriptor invalid header!", self.filter_id)
                        return
                    if line_frame_id == self.last_frame_id:
                        if wrote_last_frame_id is False:
                            wrote_last_frame_id = True
                            f_out.write(previous_points)
                    if line_frame_id == self.frame_id:
                        if wrote_frame_id is False:
                            wrote_frame_id = True
                            f_out.write(current_points)
                    if line_frame_id != self.frame_id and \
                            line_frame_id != self.last_frame_id:
                        f_out.write(line)
            if wrote_last_frame_id is False:
                f_out.write(previous_points)
            if wrote_frame_id is False:
                f_out.write(current_points)

        os.remove(file_copy_path)
        if self.draw_enabled:
            self.on_click_button_draw()
        self.update_save_button(False)

    # Methods ----------------------------------------------------------------------------------------------------------
    def redraw_line(self, point):
        point_id = point.get_item_id()
        prev_point = None

        if point_id in self.point_items.keys():
            points = self.point_items[point_id]
            if len(points) < 2:
                return

            prev_point = points[0]
            point      = points[1]

        if prev_point is None or point is None:
            return

        if point_id in self.lines.keys():
            self.scene.removeItem(self.lines[point_id])
            del self.lines[point_id]

        line = PointsTrackingLineItem(point_id,
                                      int(point.pos().x())      + self.diameter // 2,
                                      int(point.pos().y())      + self.diameter // 2,
                                      int(prev_point.pos().x()) + self.diameter // 2,
                                      int(prev_point.pos().y()) + self.diameter // 2)
        line_pen = QPen(QColor(255, 0, 0, 100))
        line_pen.setWidth(self.line_width)
        line.setPen(line_pen)
        line.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.lines[point_id] = line
        self.scene.addItem(line)
        self.on_main_splitter_moved()
        self.edit_finish = True

    def set_color_dict(self, nr):
        for i in range(nr):
            self.color_dict[i] = QColor(random.choice(range(0, 256, 1)), random.choice(range(0, 256, 1)),
                                        random.choice(range(0, 256, 1)))

    def load_points(self, items, prev_items=None):
        self.show_number_of_items(len(items))
        offset_x = 0
        if prev_items:
            offset_x = self.offset_x

        for point in items:
            point_prev_x     = None
            point_prev_y     = None
            point_prev_score = None

            if prev_items:
                for point_prev in prev_items:
                    if int(point[2]) == int(point_prev[2]):
                        point_prev_x     = int(point_prev[0] + offset_x)
                        point_prev_y     = int(point_prev[1])
                        point_prev_score = int(point_prev[3])

            self.draw_point(point_id =int(point[2])  ,
                            point_x=int(point[0])    , point_y=int(point[1])    , point_score=int(point[3]),
                            point_prev_x=point_prev_x, point_prev_y=point_prev_y, point_prev_score=point_prev_score)

        self.on_main_splitter_moved()
        self.last_item_id = len(self.items)

    def get_item_details(self, point, prev=False):
        offset_x = 0
        frame_id = self.frame_id
        if prev:
            offset_x = self.offset_x
            frame_id = self.last_frame_id

        return "{},{},{},{},{}\n".format(frame_id,
                                         int(point.pos().x()) - offset_x,
                                         int(point.pos().y()),
                                         int(point.get_item_id()),
                                         int(point.score))

    def draw_point(self, point_id, point_x, point_y, point_score, point_prev_x, point_prev_y, point_prev_score):
        if not point_id in self.color_dict.keys():
            self.color_dict[point_id] = \
                QColor(random.choice(range(0, 256, 1)), random.choice(range(0, 256, 1)),
                       random.choice(range(0, 256, 1)))

        if point_id not in self.point_items.keys():
            self.point_items[point_id] = []

        color = self.color_dict[point_id]
        if point_prev_x and point_prev_y:
            qt_point_prev = PointsTrackingItem(point_id, point_prev_score, QPen(color), QBrush(color),
                                               point_prev_x, point_prev_y, 0, 0, self.diameter, self.diameter)
            qt_point_prev.set_connection(self.on_item_changed, self.on_edit_finish)
            self.point_items[point_id].append(qt_point_prev)
            self.scene.addItem(qt_point_prev)

        qt_point = PointsTrackingItem(point_id, point_score, QPen(color), QBrush(color),
                                      point_x, point_y, 0, 0, self.diameter, self.diameter)
        qt_point.set_connection(self.on_item_changed, self.on_edit_finish)
        self.point_items[point_id].append(qt_point)
        self.scene.addItem(qt_point)
        self.redraw_line(qt_point)
        return qt_point

    def remove_point(self, deleted_point):
        point_id = deleted_point.get_item_id()
        if point_id in self.point_items.keys():
            point_list = self.point_items[point_id]
            for point in point_list:
                if point == deleted_point:
                    self.scene.removeItem(point)
                    self.point_items[point_id].remove(deleted_point)

            if point_id in self.lines.keys():
                self.scene.removeItem(self.lines[point_id])
                del self.lines[point_id]

    def remove_line(self, deleted_line):
        line_id = deleted_line.get_item_id()
        if line_id in self.point_items.keys():
            point_list = self.point_items[line_id]

            for point in point_list:
                self.scene.removeItem(point)

            self.point_items[line_id].clear()
            del self.point_items[line_id]

            if line_id in self.lines.keys():
                self.scene.removeItem(self.lines[line_id])
                del self.lines[line_id]

    def delete_point(self, item):
        self.update_save_button(True)

        if isinstance(item, PointsTrackingItem):
            self.remove_point(item)
        else:
            self.remove_line(item)

        self.show_number_of_items(len(self.items))

    def new_item_id_is_valid(self):
        new_id = self.qtPointID.value()
        for point in self.items:
            if point.get_item_id() == new_id:
                return False
        return True

    # Inherited Slots --------------------------------------------------------------------------------------------------
    def on_load_data(self, line_index, ts_sync_timestamp):
        self.last_frame_id = self.frame_id
        self.show_runtime_values(frame_id=line_index - 1)
        data, img = super().on_load_data(line_index, ts_sync_timestamp)
        self.clear()

        if img is None or data is None:
            return

        ts_stop_pts, sampling_time_pts, points = data
        ts_stop, sampling_time, ts_image, cv_image, _ = img

        self.show_runtime_values(ts_stop=ts_stop_pts, ts_sampling_time=sampling_time,
                                 ts_image=ts_image, frame_id=self.frame_id)

        if self.prev_image is not None:
            concat_image = cv2.hconcat([cv_image, self.prev_image])
            self.load_image(concat_image)
        else:
            self.load_image(cv_image)

        if self.frame_id == 0:
            self.set_color_dict(len(points))

        if points is not None:
            if self.prev_points is not None:
                self.load_points(points, self.prev_points)
            else:
                self.load_points(points)

            self.prev_points = points.copy()
            self.prev_image = cv_image.copy()
        else:
            self.signal_warning_signal.emit("Points NONE for image filter id {}, line index {}, timestamp sync {}!"
                                            .format(self.filter_id_image, line_index, ts_sync_timestamp),
                                            self.filter_id)

    def on_click_button_delete(self):
        if len(self.scene.selectedItems()) > 0:
            point = self.scene.selectedItems()[0]
            if isinstance(point, PointsTrackingItem):
                self.changes.add_item(ChangesItem(self.get_item_details(point), was_removed=True))
            self.delete_point(point)

    # Slots ------------------------------------------------------------------------------------------------------------
    def on_copy_paste_points(self):
        if self.copy_points is None:
            self.copy_points = self.items.copy()
            self.qtButtonCopyPoints.setText("Paste Points")
        else:
            if len(self.items) == 0:
                for point in self.copy_points:
                    self.draw_point(int(point.get_item_id()), int(point.pos().x()),
                                    int(point.pos().y()), int(point.score))
                self.last_item_id = len(self.items)

            self.copy_points = None
            self.qtButtonCopyPoints.setText("Copy Points")
            self.update_save_button(True)

    def on_draw_new_point(self, x, y):
        self.draw_point(point_id=self.last_item_id, point_x=x, point_y=y, point_score=1)
        self.last_item_id = len(self.items) + 1
        self.update_save_button(True)

    def on_draw_points(self):
        self.scene.on_active_draw()

        if self.scene.activate:
            self.qtButtonDrawPoints.setText("Stop drawing")
            self.last_item_id = len(self.items) + 1
        else:
            self.qtButtonDrawPoints.setText("Draw Points")

    def on_display_cursor_position(self, x, y):
        self.qtXCursor.setText(str(x))
        self.qtYCursor.setText(str(y))

    def on_display_item_id(self):
        self.qtPointID.valueChanged.disconnect()
        self.qtPointID.setValue(-1)
        self.qtPointID.setEnabled(False)
        self.show_item_id()

        if self.scene.selectedItems().__len__() > 0:
            item = self.scene.selectedItems()[0]
            self.show_item_id(item.get_item_id())
            self.qtPointID.setEnabled(True)
            self.qtPointID.setValue(item.get_item_id())

        self.qtPointID.valueChanged.connect(self.on_change_point_id)

    def on_change_point_id(self):
        if len(self.scene.selectedItems()) > 0:
            if not self.new_item_id_is_valid():
                self.signal_error_signal.emit("Point ID {} already exists! Filter {}!"
                                              .format(self.qtPointID.value(), self.filter_id), self.filter_id)
                return
            point = self.scene.selectedItems()[0]
            old_id = point.get_item_id()
            point.set_item_id(self.qtPointID.value())
            self.show_item_id(self.qtPointID.value())
            self.changes.add_item(ChangesItem(self.get_item_details(point), old_id=old_id))
            self.update_save_button(True)
        else:
            self.qtPointID.setValue(-1)

    # Inherited Slots -------- CTRL + C / X / V / Z --------------------------------------------------------------------
    def on_copy(self):
        if len(self.scene.selectedItems()) > 0:
            point = self.scene.selectedItems()[0]
            self.clipboard = ["copy", self.get_item_details(point)]
            self.update_save_button(True)

    def on_cut(self):
        if len(self.scene.selectedItems()) > 0:
            point = self.scene.selectedItems()[0]
            self.clipboard = ["cut", self.get_item_details(point)]
            self.remove_point(point)
            self.update_save_button(True)

    def on_paste(self):
        if self.clipboard is None:
            return

        point = self.clipboard[1].replace('\n', '').split(',')

        frame_id = int(point[0])
        item_x = int(point[1])
        item_y = int(point[2])
        item_id = int(point[3])
        item_s = int(point[4])

        if self.clipboard[0] == "copy" or frame_id != self.frame_id:
            self.last_item_id += 1
            item_id = self.last_item_id

        point = self.draw_point(point_id=item_id, point_x=item_x, point_y=item_y, point_score=item_s)
        self.changes.add_item(ChangesItem(self.get_item_details(point), is_new=True))
        self.clipboard = None
        self.show_number_of_items(self.items.__len__())

    def on_undo(self, item):
        if self.clipboard is None:
            self.clipboard = None

        details = str(item.item)
        point = details.replace('\n', '').split(',')
        frame_id = int(point[0])
        item_x = int(point[1])
        item_y = int(point[2])
        item_id = int(point[3])
        item_s = int(point[4])

        if self.frame_id != frame_id:
            self.signal_warning_signal.emit("Undo not possible because of different frame id {} != {}!"
                                            .format(self.frame_id, frame_id), self.filter_id)
            return

        if item.was_removed:
            self.draw_point(point_id=item_id, point_x=item_x, point_y=item_y, point_score=item_s)

        point = self.find_item_by_id(item_id)
        if point is not None:
            self.remove_point(point)
            if item.old_id != -1:
                self.draw_point(point_id=item_id, point_x=item_x, point_y=item_y, point_score=item_s)
            else:
                if not item.is_new:
                    self.draw_point(point_id=item_id, point_x=item_x, point_y=item_y, point_score=item_s)

        self.scene.update()
        self.scene.clearSelection()
        self.show_number_of_items(self.items.__len__())
