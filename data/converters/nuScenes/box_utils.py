"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

import numpy as np
from typing import List, Tuple
from pyquaternion import Quaternion

global checked_samples
checked_samples = []


class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None,
                 visibility_token: int = -1):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token
        self.visibility_token = visibility_token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def get_2D_format(self, wlh_factor: float = 1.0) -> list:
        """
        Returns the x y x w h l parameters.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <list: 1, 6>. X Y Z and W H L of the label
        """
        w, l, h = self.wlh * wlh_factor
        x, y, z = self.center

        return [self.name, x, y, z, w, h, l]

    def get_3D_format(self, wlh_factor: float = 1.0) -> list:
        """
        Returns the x y x w h l roll pitch yaw parameters.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <list: 1, 9>. X Y Z W H L ROLL PITCH YAW for the label
        """
        w, l, h = self.wlh * wlh_factor
        x, y, z = self.center
        roll, pitch, yaw = self.orientation.yaw_pitch_roll

        return [self.name, x, y, z, w, h, l, roll, pitch, yaw]

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def to_array(self) -> np.array:
        return [self.name, self.corners()]

    def set_visibility_token(self, visibility_token: int) -> None:
        """
        Sets the degree of visibility of the box on a scale from 1(poor) to 4(best).
        :param visibility_token: degree of visibility
        :return:
        """
        self.visibility_token = visibility_token


def get_box(sample) -> Box:
    """
            Instantiates a Box class from a sample annotation record.
            :param sample_annotation_token: Unique sample_annotation identifier.
            """
    return Box(sample['translation'], sample['size'], Quaternion(sample['rotation']),
               name=sample['category_name'], token=sample['token'], visibility_token=-1)


def get_sample_filename_by_timestamp(json_object, timestamp):
    for sample in json_object:
        if sample['timestamp'] == timestamp and 'CAM' in sample['filename']:
            return sample['filename']


def get_calib_token(json_object, sample_token, timestamp):
    for sample in json_object:
        if sample["sample_token"] == sample_token and sample['timestamp'] == timestamp and "CAM" in sample["filename"]:
            return sample["calibrated_sensor_token"]


def get_camera_intrinsic(json_object, calibrated_sensor_token):
    for sample in json_object:
        if sample["token"] == calibrated_sensor_token:
            return sample["camera_intrinsic"]


def get_visibility_token(json_object, sample_token):
    for sample in json_object:
        if sample["token"] == sample_token:
            return sample["visibility_token"]


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
        all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

# search for a scene token in a json object and return its data
def group_by_sample_token(json_object, what_to_search, search_id):  # sample_annotations, sample_token
    boxes = list()
    for sample in json_object:
        if sample[search_id] == what_to_search:
            boxes.append(get_box(sample))
    return boxes


def find_sample(json_object, what_to_search):
    for sample in json_object:
        if sample['token'] == what_to_search:
            return sample


def find_sample_generic(json_object, what_to_search, search_id):
    for sample in json_object:
        if sample[str(search_id)] == what_to_search:
            if 'CAM' in sample['filename']:
                print(sample["filename"])
                return sample["filename"], sample["sample_token"]


def get_boxes(sample_all_data, sample_summary_data, annotations_data, sample_token):
    # Retrieve sensor & pose records
    boxes = []
    boxes_camera = {
        'CAM_FRONT': [], 'CAM_FRONT_RIGHT': [], 'CAM_FRONT_LEFT': [], 'CAM_BACK': [], 'CAM_BACK_LEFT': [], 'CAM_BACK_RIGHT': []
    }
    sd_record = find_sample(sample_all_data, sample_token)
    curr_sample_record = find_sample(sample_summary_data, sd_record['sample_token'])

    if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
        # If no previous annotations available, or if sample_data is keyframe just return the current ones.
        sample = find_sample(annotations_data, curr_sample_record["anns"][0])
        sample_filename, s_t = find_sample_generic(sample_all_data, sample["sample_token"], "sample_token")
        if s_t not in checked_samples:
            checked_samples.append(s_t)
            if 'CAM_FRONT_RIGHT' in sample_filename:
                boxes_camera['CAM_FRONT_RIGHT'].append(group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
            elif 'CAM_FRONT_LEFT' in sample_filename:
                boxes_camera['CAM_FRONT_LEFT'].append(group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
            elif 'CAM_FRONT' in sample_filename:
                boxes_camera['CAM_FRONT'].append(group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
            elif 'CAM_BACK_RIGHT' in sample_filename:
                boxes_camera['CAM_BACK_RIGHT'].append(group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
            elif 'CAM_BACK_LEFT' in sample_filename:
                boxes_camera['CAM_BACK_LEFT'].append(group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
            elif 'CAM_BACK' in sample_filename:
                boxes_camera['CAM_BACK'].append(group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
        boxes.append(get_box(sample))

    else:
        prev_sample_record = find_sample(sample_summary_data, curr_sample_record['prev'])

        curr_ann_recs = [find_sample(annotations_data, token) for token in curr_sample_record['anns']]
        prev_ann_recs = [find_sample(annotations_data, token) for token in prev_sample_record['anns']]

        # Maps instance tokens to prev_ann records
        prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

        t0 = prev_sample_record['timestamp']
        t1 = curr_sample_record['timestamp']
        t = sd_record['timestamp']

        # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
        t = max(t0, min(t1, t))

        boxes = []
        for curr_ann_rec in curr_ann_recs:

            if curr_ann_rec['instance_token'] in prev_inst_map:
                # If the annotated instance existed in the previous frame, interpolate center & orientation.
                prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

                # Interpolate center.
                center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                             curr_ann_rec['translation'])]

                # Interpolate orientation.
                rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                            q1=Quaternion(curr_ann_rec['rotation']),
                                            amount=(t - t0) / (t1 - t0))

                box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                          token=curr_ann_rec['token'])
            else:
                # If not, simply grab the current annotation.
                sample = find_sample(annotations_data, curr_ann_rec['token'])
                sample_filename, s_t = find_sample_generic(sample_all_data, sample["sample_token"], "sample_token")
                if s_t not in checked_samples:
                    checked_samples.append(s_t)
                    if 'CAM_FRONT_RIGHT' in sample_filename:
                        boxes_camera['CAM_FRONT_RIGHT'].append(
                            group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
                    elif 'CAM_FRONT_LEFT' in sample_filename:
                        boxes_camera['CAM_FRONT_LEFT'].append(
                            group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
                    elif 'CAM_FRONT' in sample_filename:
                        boxes_camera['CAM_FRONT'].append(
                            group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
                    elif 'CAM_BACK_RIGHT' in sample_filename:
                        boxes_camera['CAM_BACK_RIGHT'].append(
                            group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
                    elif 'CAM_BACK_LEFT' in sample_filename:
                        boxes_camera['CAM_BACK_LEFT'].append(
                            group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
                    elif 'CAM_BACK' in sample_filename:
                        boxes_camera['CAM_BACK'].append(
                            group_by_sample_token(annotations_data, sample[s_t], "sample_token"))
                box = get_box(sample)

            boxes.append(box)
    return boxes, boxes_camera