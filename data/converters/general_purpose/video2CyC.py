"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

"""
Script for converting video to CyberCortex.AI DataBase
"""

from data.converters.CyC_DatabaseFormat import CyC_DataBase
import time
import argparse
import cv2


def convert_data(args):
    # Set video capture
    cap = cv2.VideoCapture(args.input_path)
    ret, frame = cap.read()
    if not ret:
        print('Video not available. Exiting..')
        return

    # Create empty database
    db = CyC_DataBase(db_path=args.output_path, core_id=1)
    db.add_stream(filter_type='image', filter_id=1, filter_name='CameraFront')
    # db.show_packing_info()
    db.create_db()

    # Prepare timestamp
    # ts_start = int(time.time())
    ts_start = 1631179572
    ts_stop = ts_start
    sampling_time = 5

    # Receive data
    idx = 0
    count = args.num
    while ret:
        # Data to save
        cap.set(cv2.CAP_PROP_POS_MSEC, ((idx + int(args.start)) * 60))
        ret, frame = cap.read()
        if args.size[0] != -1 and args.size[1] != -1:
            frame = cv2.resize(frame, (args.size[1], args.size[0]))

        # Pack data
        data = {'datastream_1': {  # image
            'name': '{}.jpg'.format(ts_stop),
            'image': frame
            }
        }

        # Add data to database
        db.add_data(ts_start=ts_start, ts_stop=ts_stop, data=data)

        # Advance timestamp
        ts_start = ts_stop
        ts_stop += sampling_time

        # Print progress
        idx += 1
        if idx % 2 == 0:
            if count != -1:
                print('Done {0} out of {1}.'.format(idx, count))
            else:
                print('Done {0}.'.format(idx))
        if count != -1 and idx >= count:
            break

    print('Finished process.')


def view_data(args):
    # Set video capture
    cap = cv2.VideoCapture(args.input_path)
    ret, frame = cap.read()
    if not ret:
        print('Video not available. Exiting..')
        return

    idx = 0
    count = args.num
    while ret:
        # Data to save
        cap.set(cv2.CAP_PROP_POS_MSEC, ((idx + int(args.start)) * 60))
        ret, frame = cap.read()
        if args.size[0] != -1 and args.size[1] != -1:
            frame = cv2.resize(frame, (args.size[1], args.size[0]))

        # Print info
        print('Image {}:'.format(idx))
        print('Shape: ' + str(frame.shape))
        print('\n')

        # Show image
        cv2.imshow('frame', frame)
        cv2.setWindowTitle("frame", 'Frame-{0}'.format(idx))
        cv2.waitKey(0)

        # Progress
        idx += 1
        if count != -1 and idx >= count:
            break


def main():
    parser = argparse.ArgumentParser(description='Useful script for converting random data to CyberCortex.AI format')

    parser.add_argument('--input_path', '-i', default=r'D:\dev\Adnotation\driving_video.mp4',
                        help='Path to the video')
    parser.add_argument('--output_path', '-o', default=r'D:\dev\Adnotation\driving_dataset',
                        help='Location where the CyberCortex.AI DataBase will be saved')

    parser.add_argument('--start', '-s',   default=0,
                        help='Starting second for converting video')
    parser.add_argument('--num', '-n',  default=1758,
                        help='Number of frames to retrieve. Set to -1 to convert all the video.')
    parser.add_argument('--size',  default=[-1, -1],
                        help='Resize all images to this size. Set to -1 to not resize.')

    parser.add_argument('--mode', '-m', default='view', help='view / convert')

    args = parser.parse_args()

    if args.mode == 'convert':
        convert_data(args=args)
    elif args.mode == 'view':
        view_data(args=args)
    else:
        print('Mode {} does not exist. Try \'view\' or \'convert\'.'.format(parser.parse_args().mode))


if __name__ == '__main__':
    main()
