"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from tools.Annotation.items.DojoGraphicItem import DojoGraphicItem

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Roi2DItem(QGraphicsRectItem, DojoGraphicItem):
    # Handles ----------------------------------------------------------------------------------------------------------
    handleTopLeft = 1
    handleTopMiddle = 2
    handleTopRight = 3
    handleMiddleLeft = 4
    handleMiddleRight = 5
    handleBottomLeft = 6
    handleBottomMiddle = 7
    handleBottomRight = 8
    handleSize = +8.0
    handleSpace = -8.0

    # Cursors ----------------------------------------------------------------------------------------------------------
    handleCursors = {
        handleTopLeft: Qt.SizeFDiagCursor,
        handleTopMiddle: Qt.SizeVerCursor,
        handleTopRight: Qt.SizeBDiagCursor,
        handleMiddleLeft: Qt.SizeHorCursor,
        handleMiddleRight: Qt.SizeHorCursor,
        handleBottomLeft: Qt.SizeBDiagCursor,
        handleBottomMiddle: Qt.SizeVerCursor,
        handleBottomRight: Qt.SizeFDiagCursor,
    }

    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(self, roi_id, roi_class, pen, *args):
        super().__init__(*args)

        self.handles = {}

        self.item_id = roi_id
        self.roi_class = roi_class
        self.pen = pen

        self.handleSelected = None
        self.mousePressPos = None
        self.mousePressRect = None

        self.actions = ""
        self.locs = ""

        self.setAcceptHoverEvents(True)
        self.set_tags()
        self._update_handles_pos()

    # Methods ----------------------------------------------------------------------------------------------------------
    def set_class(self, roi_class):
        self.roi_class = roi_class

    def get_class(self):
        return self.roi_class

    # Private Methods --------------------------------------------------------------------------------------------------
    def _update_handles_pos(self):
        """
        Update current resize handles according to the shape size and position.
        """
        s = self.handleSize
        o = self.handleSize + self.handleSpace
        b = self.rect().adjusted(-o, -o, o, o)
        self.handles[self.handleTopLeft] = QRectF(b.left(), b.top(), s, s)
        self.handles[self.handleTopMiddle] = QRectF(b.center().x() - s / 2, b.top(), s, s)
        self.handles[self.handleTopRight] = QRectF(b.right() - s, b.top(), s, s)
        self.handles[self.handleMiddleLeft] = QRectF(b.left(), b.center().y() - s / 2, s, s)
        self.handles[self.handleMiddleRight] = QRectF(b.right() - s, b.center().y() - s / 2, s, s)
        self.handles[self.handleBottomLeft] = QRectF(b.left(), b.bottom() - s, s, s)
        self.handles[self.handleBottomMiddle] = QRectF(b.center().x() - s / 2, b.bottom() - s + 0.5, s, s)
        self.handles[self.handleBottomRight] = QRectF(b.right() - s, b.bottom() - s, s, s)

    def _interactive_resize(self, pos):
        """
        Perform shape interactive resize.
        """
        if not self.isSelected():
            pass

        self.signals.signal_item_changed.emit(int(self.sceneBoundingRect().x()),
                                              int(self.sceneBoundingRect().y()),
                                              int(self.sceneBoundingRect().width()),
                                              int(self.sceneBoundingRect().height()))

        offset = self.handleSize + self.handleSpace
        bounding_rect = self.boundingRect()
        rect = self.rect()
        diff = QPointF(0, 0)
        self.prepareGeometryChange()

        if self.handleSelected == self.handleTopLeft:
            from_x = self.mousePressRect.left()
            from_y = self.mousePressRect.top()
            to_x = from_x + pos.x() - self.mousePressPos.x()
            to_y = from_y + pos.y() - self.mousePressPos.y()
            diff.setX(to_x - from_x)
            diff.setY(to_y - from_y)
            bounding_rect.setLeft(to_x)
            bounding_rect.setTop(to_y)
            rect.setLeft(bounding_rect.left() + offset)
            rect.setTop(bounding_rect.top() + offset)
            self.setRect(rect)

        elif self.handleSelected == self.handleTopMiddle:
            from_y = self.mousePressRect.top()
            to_y = from_y + pos.y() - self.mousePressPos.y()
            diff.setY(to_y - from_y)
            bounding_rect.setTop(to_y)
            rect.setTop(bounding_rect.top() + offset)
            self.setRect(rect)

        elif self.handleSelected == self.handleTopRight:
            from_x = self.mousePressRect.right()
            from_y = self.mousePressRect.top()
            to_x = from_x + pos.x() - self.mousePressPos.x()
            to_y = from_y + pos.y() - self.mousePressPos.y()
            diff.setX(to_x - from_x)
            diff.setY(to_y - from_y)
            bounding_rect.setRight(to_x)
            bounding_rect.setTop(to_y)
            rect.setRight(bounding_rect.right() - offset)
            rect.setTop(bounding_rect.top() + offset)
            self.setRect(rect)

        elif self.handleSelected == self.handleMiddleLeft:
            from_x = self.mousePressRect.left()
            to_x = from_x + pos.x() - self.mousePressPos.x()
            diff.setX(to_x - from_x)
            bounding_rect.setLeft(to_x)
            rect.setLeft(bounding_rect.left() + offset)
            self.setRect(rect)

        elif self.handleSelected == self.handleMiddleRight:
            from_x = self.mousePressRect.right()
            to_x = from_x + pos.x() - self.mousePressPos.x()
            diff.setX(to_x - from_x)
            bounding_rect.setRight(to_x)
            rect.setRight(bounding_rect.right() - offset)
            self.setRect(rect)

        elif self.handleSelected == self.handleBottomLeft:
            from_x = self.mousePressRect.left()
            from_y = self.mousePressRect.bottom()
            to_x = from_x + pos.x() - self.mousePressPos.x()
            to_y = from_y + pos.y() - self.mousePressPos.y()
            diff.setX(to_x - from_x)
            diff.setY(to_y - from_y)
            bounding_rect.setLeft(to_x)
            bounding_rect.setBottom(to_y)
            rect.setLeft(bounding_rect.left() + offset)
            rect.setBottom(bounding_rect.bottom() - offset)
            self.setRect(rect)

        elif self.handleSelected == self.handleBottomMiddle:
            from_y = self.mousePressRect.bottom()
            to_y = from_y + pos.y() - self.mousePressPos.y()
            diff.setY(to_y - from_y)
            bounding_rect.setBottom(to_y)
            rect.setBottom(bounding_rect.bottom() - offset)
            self.setRect(rect)

        elif self.handleSelected == self.handleBottomRight:
            from_x = self.mousePressRect.right()
            from_y = self.mousePressRect.bottom()
            to_x = from_x + pos.x() - self.mousePressPos.x()
            to_y = from_y + pos.y() - self.mousePressPos.y()
            diff.setX(to_x - from_x)
            diff.setY(to_y - from_y)
            bounding_rect.setRight(to_x)
            bounding_rect.setBottom(to_y)
            rect.setRight(bounding_rect.right() - offset)
            rect.setBottom(bounding_rect.bottom() - offset)
            self.setRect(rect)

        self._update_handles_pos()

    def _display_id_info(self, painter):
        rect_w = 40
        rect_h = 25

        rect_id = QRect(self.rect().x(), self.rect().y(),
                        rect_w if self.rect().width() > rect_w else self.rect().width(),
                        rect_h if self.rect().height() > rect_h else self.rect().height())

        painter.setBrush(QBrush(self.pen.color(), Qt.SolidPattern))
        painter.drawRect(rect_id)

        # Text
        path = QPainterPath()
        font = QFont()
        font.setBold(True)
        font.setPointSize(rect_h - 5)
        painter.setPen(QPen(Qt.black, 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(Qt.white))
        painter.setFont(font)
        path.addText(rect_id.x() if self.item_id >= 10 else rect_id.x() + int(rect_id.width() / 4),
                     rect_id.y() + rect_id.height(), font, str(self.item_id) + " " + self.actions + " "
                     + self.locs)
        painter.drawPath(path)

    def _handle_at(self, point):
        """
        Returns the resize handle below the given point.
        """
        for k, v, in self.handles.items():
            if v.contains(point):
                return k
        return None

    # Inherited Events -------------------------------------------------------------------------------------------------
    def paint(self, painter, option, widget=None):
        """
        Paint the node in the graphic view.
        """
        painter.setPen(self.pen)
        painter.drawRect(self.rect())

        if not self.isSelected():
            self._display_id_info(painter)

        if self.isSelected():
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(Qt.white))
            painter.setPen(QPen(Qt.black, 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            for handle, rect in self.handles.items():
                if self.handleSelected is None or handle == self.handleSelected:
                    painter.drawEllipse(rect)

    def boundingRect(self):
        """
        Returns the bounding rect of the shape (including the resize handles).
        """
        o = 0  # self.handleSize + self.handleSpace
        return self.rect().adjusted(-o, -o, o, o)

    def itemChange(self, change, val):
        if change == QGraphicsItem.ItemPositionChange:
            self.signals.signal_item_changed.emit(int(self.sceneBoundingRect().x()),
                                                  int(self.sceneBoundingRect().y()),
                                                  int(self.sceneBoundingRect().width()),
                                                  int(self.sceneBoundingRect().height()))
        return val

    def hoverMoveEvent(self, event):
        """
        Executed when the mouse moves over the shape (NOT PRESSED).
        """
        if self.isSelected():
            handle = self._handle_at(event.pos())
            cursor = Qt.ArrowCursor if handle is None else self.handleCursors[handle]
            self.setCursor(cursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        """
        Executed when the mouse leaves the shape (NOT PRESSED).
        """
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        """
        Executed when the mouse is pressed on the item.
        """
        self.handleSelected = self._handle_at(event.pos())
        if self.handleSelected:
            self.mousePressPos = event.pos()
            self.mousePressRect = self.boundingRect()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Executed when the mouse is being moved over the item while being pressed.
        """
        if self.handleSelected is not None:
            self._interactive_resize(event.pos())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Executed when the mouse is released from the item.
        """
        super().mouseReleaseEvent(event)
        self.handleSelected = None
        self.mousePressPos = None
        self.mousePressRect = None
        self.update()
        self.signals.signal_edit_finish.emit()
