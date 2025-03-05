"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

from CyC_db_interface import CyC_FilterType, CyC_DataType, CyC_DatabaseParser


def example_read_rois(parser, filter_id, limit=-1):
    rois = parser.get_data_by_id(filter_id=filter_id)  # Filter 401 outputs 2D ROIs

    if limit > 0:
        print("Showing first {0} ROIs".format(limit))

    crt_idx = 0
    for ts_start, ts_stop, sampling_time, roi_sequence in rois:
        print("Frame number {0}".format(crt_idx + 1))
        print("ts_start={0}, ts_stop={1}, sampling_time={2}, num_rois={3}".format(
            ts_start, ts_stop, sampling_time, len(roi_sequence))
        )
        print("ROIs")
        for crt_roi in roi_sequence:
            print("frame_id={0}, roi_id={1}, cls={2}, x={3}, y={4}, width={5}, height={6}".format(
                crt_roi[0], crt_roi[1], crt_roi[2], crt_roi[3], crt_roi[4], crt_roi[5], crt_roi[6])
            )
        print("===============================================")
        crt_idx += 1
        if limit > 0:
            if crt_idx == limit:
                break


def example_read_monocamera_images(parser, filter_id, limit=-1):
    """
    MonoCamera does not have right image!
    """
    image_data = parser.get_data_by_id(filter_id=filter_id)

    if limit > 0:
        print("Showing first {0} images shapes".format(limit))

    crt_idx = 0
    for ts_start, ts_stop, sampling_time, im0, im1 in image_data:
        print("Frame number: {0}".format(crt_idx + 1))
        print("ts_start={0}, ts_stop={1}, sampling_time={2}".format(ts_start, ts_stop, sampling_time))

        # Check if image was read
        if im0 is not None:
            print("Left image shape: {0}".format(im0.shape))

        # Check if image was read
        if im1 is not None:
            print("Right image shape: {0}".format(im1.shape))
        print("===============================================")

        crt_idx += 1
        if limit > 0:
            if crt_idx == limit:
                break


def example_read_state_measurement_data(parser, filter_id, limit=-1):
    state_meas_data = parser.get_data_by_id(filter_id=filter_id)

    if limit > 0:
        print("Showing first {0} state measurement records".format(limit))

    crt_idx = 0
    for ts_start, ts_stop, sampling_time, state in state_meas_data:
        print("Frame number: {0}".format(crt_idx + 1))
        print("ts_start={0}, ts_stop={1}, sampling_time={2}, x={3}, y={4}, v={5}, yaw={6}".format(
            ts_start, ts_stop, sampling_time, state[0], state[1], state[2], state[3])
        )
        print("===============================================")
        crt_idx += 1
        if limit > 0:
            if crt_idx == limit:
                break


def example_get_image_and_rois(db_path, limit=-1):
    """
    The Images and ROIs are not synced by TIMESTAMP!!!
    """
    parser = CyC_DatabaseParser(db_path)
    filter_id_mono = 1
    filter_id_roi = 14
    monocamera_data = parser.get_data_by_id(filter_id=filter_id_mono)
    roi_data = parser.get_data_by_id(filter_id=filter_id_roi)

    if limit > 0:
        print("Showing first {0} images + rois".format(limit))

    crt_idx = 0
    for (_, _, _, im0, im1), (_, _, _, roi_sequence) in zip(monocamera_data, roi_data):
        print("Frame number {0}".format(crt_idx + 1))
        if im0 is not None:
            print("Image 0 shape: {0}".format(im0.shape))
        if im1 is not None:
            print("Image 1 shape: {0}".format(im1.shape))
        print("ROIs")
        for crt_roi in roi_sequence:
            print("frame_id={0}, roi_id={1}, cls={2}, x={3}, y={4}, width={5}, height={6}".format(
                crt_roi[0], crt_roi[1], crt_roi[2], crt_roi[3], crt_roi[4], crt_roi[5], crt_roi[6]))
        print("===============================================")
        crt_idx += 1
        if limit > 0:
            if crt_idx == limit:
                break


def simple_exercises(db_path, limit=-1):
    parser = CyC_DatabaseParser(db_path)
    example_read_rois(parser=parser, filter_id=14, limit=limit)
    example_read_monocamera_images(parser=parser, filter_id=1, limit=limit)
    example_read_state_measurement_data(parser=parser, filter_id=13, limit=limit)


if __name__ == "__main__":
    """
    READ THIS!
    ====================================================================================================================
    CyberCortex.AI Database Parser example section.
    This serves as demonstrator for lazy-iterating over CyberCortex.AI records.
    The generators returned by get_data_x_y(args) yield data on-the-fly, so no data is stored.
    If you want to use your data, you have to store it
    e.g.
    images = list()
    for ts_start, ts_stop, sampling_time, im0, im1 in monocamera_generator:
        images.append(im0)
    
    Because of generator style classes, rewinding is NOT supported. Therefore, something like:
    
    for ts_start, ts_stop, sampling_time, im0, im1 in monocamera_generator:
        print(ts_start, ...)
    ....
    for ts_start, ts_stop, sampling_time, im0, im1 in monocamera_generator:
        print(ts_start, ...)
    
    WILL NOT WORK!
    
    If you want to parse a database entry multiple times, you will have to re-initialize the database parser object
    
    ====================================================================================================================
    """

    db_path = "fake_dataset"
    limit = 5
    simple_exercises(db_path, limit)
    example_get_image_and_rois(db_path, limit)
