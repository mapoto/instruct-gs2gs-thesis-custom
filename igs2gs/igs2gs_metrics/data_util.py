import json
from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch
import plyfile


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


def read_colors_from_image(image_path) -> np.ndarray:
    source_color_image = Image.open(image_path)
    source_color_image = np.asarray(source_color_image)

    colors_s = source_color_image.reshape(-1, 3) / 255.0  # Normalize RGB to [0, 1]
    return colors_s


def read_depth(depth_path):
    depth_map = torch.load(depth_path, weights_only=True).detach().cpu()
    maxima = torch.max(depth_map)
    mask = depth_map == maxima
    depth_map[mask] = 0
    return depth_map


def load_ply(file_path):
    # Load the PLY file
    ply_data = plyfile.PlyData.read(file_path)

    # Extract the vertices
    vertices = np.vstack([ply_data["vertex"]["x"], ply_data["vertex"]["y"], ply_data["vertex"]["z"]]).T

    return vertices
