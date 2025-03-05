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
 * main_waypoints_planner.py
 *
 *  Created on: 02.11.2021
 *      Author: Sorin Grigorescu
"""

from absl import app
import os
import numpy as np
import csv
from dataclasses import dataclass
from typing import List


@dataclass
class Waypoint:
    translation:    np.array((3,), dtype=float)
    rotation:       np.array((1,), dtype=float)
    visible:        bool = True


@dataclass
class Landmark:
    id:             int
    translation:    np.array((3,), dtype=float)
    rotation:       np.array((3,), dtype=float)
    waypoints:      List[Waypoint]
    travel_time:    float = 0.0


class WaypointsPlanner:
    def __init__(self):
        self.waypoints_file = None
        self.landmarks = []

    def clearLandmarks(self):
        self.landmarks.clear()

    def loadLandmarks(self, filename: str):
        if not os.path.isfile(filename):
            print("Waypoints file ( {} ) does not exists.".format(filename))
            return

        # Parse map file
        csv_reader = csv.DictReader(open(filename, "r"))

        for row in csv_reader:
            try:
                landmark = Landmark(int(row["landmark_id"]),
                                    np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                                    np.array([float(row["roll"]), float(row["pitch"]), float(row["yaw"])]),
                                    self.str2waypoints(str(row["waypoints"])),
                                    float(row["travel_time"]))
            except:
                landmark = Landmark(int(row["landmark_id"]),
                                    np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                                    np.array([float(row["qx"]), float(row["qy"]), float(row["qz"])]),
                                    self.str2waypoints(str(row["waypoints"])),
                                    float(row["travel_time"]))

            self.landmarks.append(landmark)

    def saveWaypoints(self, waypoints_file: str, use_quaternions=True):
        writer = csv.DictWriter(open(waypoints_file, "w+", newline=''),
                                ["name", "landmark_id", "x", "y", "z", "roll", "pitch", "yaw", "travel_time", "waypoints"]
                                if not use_quaternions else
                                ["name", "landmark_id", "x", "y", "z", "qx", "qy", "qz", "qw", "travel_time", "waypoints"])
        writer.writeheader()
        mission_idx = 0
        for landmark in self.landmarks:
            if len(landmark.waypoints) == 0:
                waypoints_txt = "[[{};{};{};{}]]".format(
                    landmark.translation[0],
                    landmark.translation[1],
                    landmark.translation[2],
                    landmark.rotation[2])
            else:
                

                waypoints_str = [map(lambda v: str(v),
                                     np.concatenate(
                                         (w.translation, np.deg2rad(w.rotation))
                                     ).tolist())
                                 for w in landmark.waypoints]

                waypoints_vec = [";".join(w) for w in waypoints_str]
                waypoints_txt = "[[" + "][".join(waypoints_vec) + "]]"

            if use_quaternions:
                # http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm
                c1 = np.cos(np.deg2rad(landmark.rotation[2]) / 2)
                s1 = np.sin(np.deg2rad(landmark.rotation[2]) / 2)
                c2 = np.cos(np.deg2rad(landmark.rotation[1]) / 2)
                s2 = np.sin(np.deg2rad(landmark.rotation[1]) / 2)
                c3 = np.cos(np.deg2rad(landmark.rotation[0]) / 2)
                s3 = np.sin(np.deg2rad(landmark.rotation[0]) / 2)
                c1c2 = c1 * c2
                s1s2 = s1 * s2
                qw = c1c2 * c3 - s1s2 * s3
                qx = c1c2 * s3 + s1s2 * c3
                qy = s1 * c2 * c3 + c1 * s2 * s3
                qz = c1 * s2 * c3 - s1 * c2 * s3

                writer.writerow({"name": "Mission {}".format(mission_idx),
                                 "landmark_id": landmark.id,
                                 "x": landmark.translation[0],
                                 "y": landmark.translation[1],
                                 "z": landmark.translation[2],
                                 "qx": qx,
                                 "qy": qy,
                                 "qz": qz,
                                 "qw": qw,
                                 "travel_time": landmark.travel_time,
                                 "waypoints": waypoints_txt})
            else:
                writer.writerow({"name": "Mission {}".format(mission_idx),
                                 "landmark_id": landmark.id,
                                 "x": landmark.translation[0],
                                 "y": landmark.translation[1],
                                 "z": landmark.translation[2],
                                 "roll": np.deg2rad(landmark.rotation[0]),
                                 "pitch": np.deg2rad(landmark.rotation[1]),
                                 "yaw": np.deg2rad(landmark.rotation[2]),
                                 "travel_time": landmark.travel_time,
                                 "waypoints": waypoints_txt})
            mission_idx += 1

    @staticmethod
    def str2waypoints(str_waypoints):
        waypoints = list()
        str_waypoints = str_waypoints.split("][")
        for str_waypoint in str_waypoints:
            str_waypoint = str_waypoint.replace("[[", "")
            str_waypoint = str_waypoint.replace("]]", "")
            values = list(map(lambda x: float(x), str_waypoint.split(";")))

            is_visible = True
            if len(values) > 4:
                is_visible = values[4]

            waypoint = Waypoint(np.array(values[:3]), np.array([values[3]]), is_visible)
            waypoints.append(waypoint)

        return waypoints


def tu_WaypointsPlanner(_argv):
    planner = WaypointsPlanner()


if __name__ == '__main__':
    try:
        app.run(tu_WaypointsPlanner)
    except SystemExit:
        pass
