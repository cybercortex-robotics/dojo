"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from PyQt5.QtGui import QValidator, QRegularExpressionValidator, QPixmap, QIcon
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, Qt, QRegularExpression

from tools.Annotation.FilterObject import FilterObject, FilterWidgetItem, TreeWidgetItemColumns
from tools.Annotation.dialogs.AddDatastream import AddDatastreamWindow
from tools.Annotation.dialogs.CopyDatabase import CopyDatabase
from tools.Annotation.dialogs.CreateDatabase import CreateDatabase
from tools.Annotation.dialogs.HelpWindow import HelpWindow
from data.CyC_db_interface import CyC_DatabaseParser

from global_config import *
import threading
import os.path
import glob
import time
import csv

CFG = cfg
current_directory = os.path.dirname(__file__)
ui_directory = os.path.join(current_directory, "ui")
res_directory = os.path.join(current_directory, "res")


class MainControlWindow(QWidget):
    signal_set_play_status = pyqtSignal(bool)
    signal_show_data = pyqtSignal()

    def __init__(self, main_widget):
        super().__init__()
        uic.loadUi(os.path.join(ui_directory, 'MainControlWindow.ui'), self)
        self.setWindowTitle("CyberCortex.AI Viewer")
        self.parent = main_widget

        # Widget
        self.qtNextFrame = self.findChild(QPushButton, 'qtNextFrame')
        self.qtPrevFrame = self.findChild(QPushButton, "qtPrevFrame")
        self.qtPlay = self.findChild(QPushButton, 'qtPlay')
        self.qtSpeed = self.findChild(QSpinBox, 'qtSpeed')
        self.qtTimestamp = self.findChild(QLineEdit, 'qtTimestamp')
        self.qtTimestampSlider = self.findChild(QSlider, 'qtTimestampSlider')
        self.qtLabelsDataStreams = self.findChild(QTreeWidget, 'qtLabelsDataStreams')
        self.qtViewingArea = self.findChild(QStackedWidget, 'qtViewingArea')
        self.qtMainHorizontalLayout = self.findChild(QHBoxLayout, 'qtMainHorizontalLayout')
        self.qtButtonsLayout = self.findChild(QHBoxLayout, 'qtButtonsLayout')

        # Members
        self.splitter = QSplitter(Qt.Horizontal)
        self.parser = CyC_DatabaseParser.get_instance(cfg.DB.BASE_PATH)
        self.play = False
        self.wait = False
        self.index = 0
        self.sleep = 0

        # Signals
        self.signal_show_data.connect(self.on_show_data)
        self.signal_set_play_status.connect(self.on_set_play_status)
        self.splitter.splitterMoved.connect(self.on_splitter_moved)

        self.qtTimestampSlider.valueChanged.connect(self.on_set_value_timestamp)
        self.qtTimestampSlider.sliderReleased.connect(self.on_set_value_timestamp_show)
        self.qtNextFrame.clicked.connect(self.on_click_next_frame)
        self.qtPrevFrame.clicked.connect(self.on_click_prev_frame)
        self.qtPlay.clicked.connect(self.on_click_play)
        self.qtSpeed.valueChanged.connect(self.on_speed_changed)

        validator = QRegularExpressionValidator(QRegularExpression("[1-9][0-9]+"))
        self.qtTimestamp.setValidator(validator)
        self.qtTimestamp.textChanged.connect(self.on_check_timestamp)

        # Initialization
        self.parent.on_display_info("Selected database: {}".format(CFG.DB.BASE_PATH))
        self._load_buttons()
        self._load_filters()

        self.splitter.addWidget(self.qtLabelsDataStreams)
        self.splitter.addWidget(self.qtViewingArea)
        self.qtMainHorizontalLayout.addWidget(self.splitter)
        self.on_speed_changed(self.qtSpeed.value())
        self.splitter.setSizes([100, 5000])

    # LOAD/CHANGE CyberCortex.AI DATABASE TREE VIEW #############################################################################
    def reload_db(self):
        self.parser.refresh(CFG.DB.BASE_PATH)

        self.qtLabelsDataStreams.itemClicked.disconnect()
        self.qtLabelsDataStreams.itemChanged.disconnect()
        self.qtLabelsDataStreams.itemDoubleClicked.disconnect()
        self.qtLabelsDataStreams.clear()

        self._load_filters()

    def _load_buttons(self):
        self.qtHelpWindow = HelpWindow()
        self.qtCopyWindow = CopyDatabase()
        self.qtCreateWindow = CreateDatabase()

        self.qtButtonsLayout.addWidget(
            self._create_icon_button("load.png", "Load Database", self.parent.on_back_to_configuration_window))
        self.qtButtonsLayout.addWidget(
            self._create_icon_button("create.png", "Create Database", lambda show_window: self.qtCreateWindow.show()))
        self.qtButtonsLayout.addWidget(
            self._create_icon_button("copy.png", "Copy Database", lambda show_window: self.qtCopyWindow.show()))

        help_button = self._create_icon_button("info.jpg", "Help Menu")
        help_menu = QMenu()
        files = glob.glob(".\\dialogs\\help_text\\*")
        for file in files:
            filter_name = (file.split('\\')[-1]).split("_help.txt")[0]
            action = QAction(filter_name, self)
            action.triggered.connect(lambda fill, path=file, name=filter_name: self.on_show_help(path, name))
            help_menu.addAction(action)

        help_button.setMenu(help_menu)
        self.qtButtonsLayout.addWidget(help_button)

    def on_show_help(self, path, name):
        self.qtHelpWindow.load_file(path, name)
        self.qtHelpWindow.show()

    def _load_filters(self):
        self.qtLabelsDataStreams.setColumnCount(4)
        self.qtLabelsDataStreams.setColumnWidth(TreeWidgetItemColumns.DELETE_ICON, 15)
        self.qtLabelsDataStreams.setColumnWidth(TreeWidgetItemColumns.FILTER_CORE, 5)
        self.qtLabelsDataStreams.setColumnWidth(TreeWidgetItemColumns.FILTER_ID, 5)
        self.qtLabelsDataStreams.setColumnWidth(TreeWidgetItemColumns.FILTER_NAME, 150)

        parent = QtWidgets.QTreeWidgetItem(self.qtLabelsDataStreams)
        parent.setText(TreeWidgetItemColumns.FILTER_CORE, "Data")
        parent.setExpanded(True)

        self.addNewDatabase = AddDatastreamWindow()
        self.qtLabelsDataStreams.setItemWidget(parent, 0,
                                               self._create_icon_button("add.png",
                                                                        "Add new filter",
                                                                        lambda show_add_window:
                                                                        self.addNewDatabase.show()))

        desc_path = os.path.join(os.path.join(current_directory, CFG.DB.BASE_PATH), "datablock_descriptor.csv")
        reader = csv.DictReader(open(desc_path, "r"))

        for row in reader:
            _filter = FilterObject(row, self, self.parser)

            child = FilterWidgetItem(_filter, parent)
            child.setToolTip(TreeWidgetItemColumns.FILTER_NAME, _filter.get_detail())
            child.setText(TreeWidgetItemColumns.FILTER_CORE, str(_filter.get_filter_core_id()))
            child.setText(TreeWidgetItemColumns.FILTER_ID, "{:03d}".format(_filter.get_filter_id()))
            child.setText(TreeWidgetItemColumns.FILTER_NAME, str(_filter.get_filter_name()))

            if _filter.get_window() is None:
                child.setDisabled(True)
            else:
                self.qtViewingArea.addWidget(_filter.get_window())

            self.qtLabelsDataStreams.setItemWidget(child, TreeWidgetItemColumns.DELETE_ICON,
                                                   self._create_icon_button("delete.jpg", "Delete filter",
                                                                            child.delete_filter))

        self.qtLabelsDataStreams.itemClicked.connect(self.on_show_window)
        self.qtLabelsDataStreams.itemChanged.connect(self.on_change_filter)
        self.qtLabelsDataStreams.itemDoubleClicked.connect(self.on_edit_filter)

    def show_first_data(self):
        self.on_show_window(self.qtLabelsDataStreams.topLevelItem(0).child(0))

    def on_show_window(self, item):
        filter_id = item.text(TreeWidgetItemColumns.FILTER_ID)
        if filter_id != "":
            window = item.get_window()

            if window is None:
                return

            self.qtViewingArea.setCurrentWidget(window)
            tss = [x[1] for x in window.timestamps]

            if len(tss):
                self.on_init_timestamp_slider(0, len(window.timestamps) - 1)

            self.signal_show_data.emit()

    @staticmethod
    def _create_icon_button(file_name, tool_tip, slot=None, size=QtCore.QSize(15, 15)):
        icon = QIcon(QPixmap(os.path.join(res_directory, file_name)))
        button = QPushButton()
        button.setIcon(icon)
        button.setIconSize(size)
        button.setFixedSize(size)
        button.setToolTip(tool_tip)

        if slot is not None:
            button.clicked.connect(slot)

        return button

    def on_edit_filter(self, item, column):
        filter_id = item.text(TreeWidgetItemColumns.FILTER_ID)
        if filter_id == "":
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            return

        if column == 0 or column == 1:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        else:
            item.setFlags(item.flags() | Qt.ItemIsEditable)

        self.qtLabelsDataStreams.editItem(item, column)

    @staticmethod
    def on_change_filter(item, column):
        if column == TreeWidgetItemColumns.FILTER_ID:
            item.change_filter_id()
        if column == TreeWidgetItemColumns.FILTER_NAME:
            item.change_filter_name()

    # LOAD/CHANGE CyberCortex.AI DATASTREAM DATA ################################################################################
    def _check_index(self):
        if self.index == len(self.qtViewingArea.currentWidget().timestamps):
            self.signal_set_play_status.emit(False)
            self.index = 0
        elif self.index < 0:
            self.index = 0
        self.on_set_value_timestamp_slider()

    def on_show_data(self):
        self._check_index()
        self.on_set_value_timestamp_slider()

        if not self.qtViewingArea.currentWidget().is_processing:
            tss = self.qtViewingArea.currentWidget().timestamps
            if len(tss):
                if self.index >= len(tss):
                    self.index = 0
                self.qtTimestamp.setText(str(tss[self.index][1]))
                self.qtViewingArea.currentWidget().on_load_data(
                    self.qtViewingArea.currentWidget().timestamps[self.index][0],
                    self.qtViewingArea.currentWidget().timestamps[self.index][2])
                self.wait = False

    # LOAD/CHANGE CyberCortex.AI DATASTREAM SPEED/SPLITTER ######################################################################
    def on_splitter_moved(self):
        self.qtViewingArea.currentWidget().on_main_splitter_moved()

    def on_speed_changed(self, value):
        self._set_sleep(int(value), self.qtSpeed.minimum(), self.qtSpeed.maximum())

    def _set_sleep(self, value, minimum, maximum):
        max_sleep_time = 1
        min_sleep_time = 0.2
        self.sleep = max_sleep_time - (max_sleep_time - min_sleep_time) / (maximum - minimum + 1) * value

    # LOAD/CHANGE CyberCortex.AI DATASTREAM PLAY/NEXT/PREV ######################################################################
    def _enable_play_menu(self, value):
        self.qtPlay.setEnabled(self.play_enabled)
        self.qtNextFrame.setEnabled(value)
        self.qtPrevFrame.setEnabled(value)
        self.qtSpeed.setEnabled(value)
        self.qtTimestamp.setEnabled(value)
        self.qtTimestampSlider.setEnabled(value)

    def _next_frame(self):
        try:
            self.qtTimestamp.textChanged.disconnect()
        except:
            pass

        while self.play:
            if not self.wait:
                self.wait = True
                self.index += 1
                self.signal_show_data.emit()
                time.sleep(self.sleep)
        self.qtTimestamp.textChanged.connect(self.on_check_timestamp)

    def on_set_play_status(self, status):
        self.play = status

        if self.play:
            self.qtPlay.setText('Pause')
        else:
            self.qtPlay.setText('Play')

    def on_click_play(self):
        self.play = not self.play
        self.signal_set_play_status.emit(self.play)

        if self.play:
            threading.Thread(target=self._next_frame).start()

    def on_click_next_frame(self):
        self.qtTimestamp.textChanged.disconnect()
        self.signal_set_play_status.emit(False)
        self.index += 1
        self.signal_show_data.emit()
        self.qtTimestamp.textChanged.connect(self.on_check_timestamp)

    def on_click_prev_frame(self):
        self.qtTimestamp.textChanged.disconnect()
        self.signal_set_play_status.emit(False)
        self.index -= 1
        self.signal_show_data.emit()
        self.qtTimestamp.textChanged.connect(self.on_check_timestamp)

    # LOAD/CHANGE CyberCortex.AI DATASTREAM SLIDER ##############################################################################
    def _check_timestamp(self):
        timestamp = self.qtTimestamp.text()
        lss = self.qtViewingArea.currentWidget().timestamps

        for index in range(len(lss)):
            if str(lss[index][1]) == timestamp:
                self.index = index
                self.qtTimestampSlider.setValue(self.index)
                self.signal_show_data.emit()
                return

        self.parent.on_display_error("Timestamp {} not found!".format(timestamp), 1)

    def on_check_timestamp(self):
        sender = self.sender()
        validator = sender.validator()
        state = validator.validate(sender.text(), 0)[0]
        color = '#f6989d'  # red
        if state == QValidator.Acceptable:
            color = '#ffff'  # '#c4df9b'  # green
            self._check_timestamp()
        elif state == QValidator.Intermediate:
            color = '#fff79a'  # yellow
        sender.setStyleSheet('QLineEdit { background-color: %s }' % color)

    def on_init_timestamp_slider(self, minimum, maximum):
        self.qtTimestampSlider.setMinimum(minimum)
        self.qtTimestampSlider.setMaximum(maximum)

    def on_set_value_timestamp_slider(self):
        self.qtTimestampSlider.setValue(self.index)

    def on_set_value_timestamp_show(self):
        self.index = self.qtTimestampSlider.value()
        self.signal_show_data.emit()

    def on_set_value_timestamp(self):
        tss = self.qtViewingArea.currentWidget().timestamps
        if len(tss):
            if self.index >= len(tss):
                self.index = 0
            self.qtTimestamp.setText(str(tss[self.index][1]))
