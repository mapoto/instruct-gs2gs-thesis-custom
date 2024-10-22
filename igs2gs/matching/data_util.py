import json
from pathlib import Path

import numpy as np


def get_frame_data(camera_frames, name):
    for frame in camera_frames:
        id = Path(frame["file_path"]).name.replace("frame_", "").replace(".png", "")
        if int(id) == int(name):
            print(frame["file_path"])

            # Extract relevant data
            width = frame["w"]
            height = frame["h"]

            # Camera intrinsics
            fx = frame["fl_x"]
            fy = frame["fl_y"]
            cx = frame["cx"]
            cy = frame["cy"]

            # View matrix
            viewmat = np.array(frame["transform_matrix"]).reshape(4, 4)

            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            return (width, height), intrinsics, viewmat
    pass


def load_transform_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def load_camera_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def parse_intrinsic(intrinsic):
    fx = intrinsic["fx"]
    fy = intrinsic["fy"]
    cx = intrinsic["cx"]
    cy = intrinsic["cy"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def parse_transform(transform):
    homogeneous = np.array(transform)
    # Add the last row
    homogeneous = np.vstack([homogeneous, [0, 0, 0, 1]])
    return homogeneous.reshape(4, 4)
