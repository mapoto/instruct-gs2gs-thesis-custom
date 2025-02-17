import numpy as np
import matplotlib.pyplot as plt
import plyfile
from igs2gs.igs2gs_metrics.data_util import load_camera_data, parse_intrinsic, parse_transform
import torch
from pathlib import Path
import cv2
from PIL import Image


def load_ply(file_path):
    # Load the PLY file
    ply_data = plyfile.PlyData.read(file_path)

    # Extract the vertices
    vertices = np.vstack([ply_data["vertex"]["x"], ply_data["vertex"]["y"], ply_data["vertex"]["z"]]).T

    return vertices


def read_colors_from_image(image_path) -> np.ndarray:
    source_color_image = Image.open(image_path)
    source_color_image = np.asarray(source_color_image)

    colors_s = source_color_image.reshape(-1, 3) / 255.0  # Normalize RGB to [0, 1]
    return colors_s


def point_cloud_csv(file_path):

    point_cloud = np.loadtxt(file_path, delimiter=",", skiprows=1)
    colors = point_cloud[:, 3:]
    points = point_cloud[:, :3]
    return points, colors


def read_depth(depth_path):
    depth_map = torch.load(depth_path, weights_only=True).detach().cpu()
    maxima = torch.max(depth_map)
    mask = depth_map == maxima
    depth_map[mask] = 0
    return depth_map


def depth_as_normalized_camera_coords(depth_image, intrinsic_matrix):
    height, width = int(intrinsic_matrix[1, 2] * 2), int(intrinsic_matrix[0, 2] * 2)

    # Depths are stored in a 1D tensor
    depth_image = depth_image.reshape(height, width)

    # Flatten the depth values
    depth = depth_image.reshape(-1)

    # Flip the depth values
    depth = depth * -1

    # Generate pixel coordinates
    i, j = np.indices((height, width))

    # Flatten the pixel coordinates
    i = i.reshape(-1)
    j = j.reshape(-1)

    # Flip the x-axis
    j = width - j

    # Convert pixel coordinates to normalized camera coordinates
    uv_coord = np.vstack((j, i) * depth)
    uv_coord = np.vstack((uv_coord, depth))

    return uv_coord


def compute_intersection(point_cloud, target_intrinsics, target_extrinsics, target_depth):
    # Convert the point cloud to homogeneous coordinates source camera
    point_cloud_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Apply the extrinsic matrix to transform the point cloud to camera coordinates
    camera_coords = np.linalg.inv(target_extrinsics) @ point_cloud_homogeneous.T

    # Apply the intrinsic matrix to project the point to image coordinates
    image_coords_homogeneous = target_intrinsics @ camera_coords[:3]

    # Normalize the coordinates to get the 2D image coordinates
    uv_coord = image_coords_homogeneous[:2] / image_coords_homogeneous[2]

    # Mirror the x-axis to match the image coordinates
    uv_coord[0] = target_intrinsics[0, 2] * 2 - uv_coord[0]

    # Depth map from the target camera
    target_depth = read_depth(depth_path=DATA_PATH / f"{target}_depth.pt")
    target_depth_uv = depth_as_normalized_camera_coords(target_depth.numpy(), target_intrinsics)
    depths = target_depth_uv[2, :]

    depth_intersection = np.zeros((np.int64(target_intrinsics[1, 2] * 2), np.int64(target_intrinsics[0, 2] * 2)))

    # check if the uv_coord is within the image size
    valid = np.logical_and(uv_coord[0] >= 0, uv_coord[0] < depth_intersection.shape[1])
    valid = np.logical_and(valid, uv_coord[1] >= 0)
    valid = np.logical_and(valid, uv_coord[1] < depth_intersection.shape[0])

    uv_coord = uv_coord[:, valid]
    depths = depths[valid]

    depth_intersection[uv_coord[1].astype(int), uv_coord[0].astype(int)] = depths * -1

    return depth_intersection


if __name__ == "__main__":

    DATA_PATH = Path(
        "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/10-16-18-18_Dora_as-if-it-were-by-modigliani_42_5.0_0.5_2.0_0.2/30000"
    )

    # Load the point cloud from source camera
    source = "18"
    point_cloud, colors18 = point_cloud_csv("/home/lucky/point_cloud_18_new.csv")

    # Target camera intrinsics and extrinsics
    target = "8"
    camera_json = DATA_PATH / f"camera_{target}.json"
    target_color_path = DATA_PATH / f"{target}_render.png"
    camera_data = load_camera_data(camera_json)

    target_intrinsics = parse_intrinsic(camera_data)
    target_extrinsics = parse_transform(camera_data["camera_to_world"])

    # Convert the point cloud to homogeneous coordinates source camera
    point_cloud_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Apply the extrinsic matrix to transform the point cloud to camera coordinates
    camera_coords = np.linalg.inv(target_extrinsics) @ point_cloud_homogeneous.T

    # Apply the intrinsic matrix to project the point to image coordinates
    image_coords_homogeneous = target_intrinsics @ camera_coords[:3]

    # Normalize the coordinates to get the 2D image coordinates
    uv_coord = image_coords_homogeneous[:2] / image_coords_homogeneous[2]

    # Mirror the x-axis to match the image coordinates
    uv_coord[0] = target_intrinsics[0, 2] * 2 - uv_coord[0]

    # Depth map from the target camera
    depth8 = read_depth(depth_path=DATA_PATH / f"{target}_depth.pt")
    depth8_uv = depth_as_normalized_camera_coords(depth8.numpy(), target_intrinsics)
    depths = depth8_uv[2, :]

    depth_intersection = np.zeros((np.int64(target_intrinsics[1, 2] * 2), np.int64(target_intrinsics[0, 2] * 2)))

    # check if the uv_coord is within the image size
    valid = np.logical_and(uv_coord[0] >= 0, uv_coord[0] < depth_intersection.shape[1])
    valid = np.logical_and(valid, uv_coord[1] >= 0)
    valid = np.logical_and(valid, uv_coord[1] < depth_intersection.shape[0])

    uv_coord = uv_coord[:, valid]
    depths = depths[valid]

    depth_intersection[uv_coord[1].astype(int), uv_coord[0].astype(int)] = depths * -1

    # Covert to black and white image
    depth_intersection_binary = np.where(depth_intersection > 0, 1, 0)

    # Store depth intersection as an image
    cv2.imwrite("/home/lucky/depth_intersection.png", depth_intersection_binary * 255)

    target_color_image = cv2.imread(target_color_path)

    # applies mask to the color image of the target camera
    colors_masked = target_color_image * depth_intersection_binary[:, :, None]

    cv2.imwrite("/home/lucky/colors_masked.png", colors_masked)

    # visualize_depth(depth_intersection)
