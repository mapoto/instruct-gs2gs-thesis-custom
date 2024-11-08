import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import torch
import json
from igs2gs.igs2gs_metrics.data_util import get_frame_data, load_transform_json
from PIL import Image
import open3d as o3d
import copy

IMG_OUT_PATH = Path("/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/")


def unproject_to_3d(depth_map, rgb_image, intrinsics):
    """
    Unprojects the RGB image and depth map into 3D world coordinates.
    """
    h, w = depth_map.shape

    # Generate pixel grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Get intrinsics
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Unproject points to 3D
    x_3d = (x - cx) * depth_map / fx
    y_3d = (y - cy) * depth_map / fy
    z_3d = depth_map

    # Combine to get 3D points
    points_3d = np.stack((x_3d, y_3d, z_3d), axis=-1)

    # Reshape points and corresponding RGB values
    points_3d = points_3d.reshape(-1, 3)  # (N, 3)
    rgb_values = rgb_image.reshape(-1, 3)  # (N, 3)

    return points_3d, rgb_values


def transform_points(points_3d, extrinsic_source, extrinsic_target):
    """
    Transforms 3D points from one camera's coordinate system to another.
    """
    # Add a fourth column for homogeneous coordinates
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Compute the relative transformation between the cameras
    relative_transform = np.linalg.inv(extrinsic_target) @ extrinsic_source

    # Apply the transformation to the 3D points
    points_3d_transformed = points_3d_homogeneous @ relative_transform.T

    return points_3d_transformed[:, :3]  # Return inhomogeneous 3D points


def reproject_to_2d(points_3d, intrinsics):
    """
    Projects 3D points into a 2D image plane using camera intrinsics.
    """
    # Apply perspective projection
    points_2d = points_3d[:, :2] / points_3d[:, 2:3]

    # Apply intrinsics
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = points_2d[:, 0] * fx + cx
    v = points_2d[:, 1] * fy + cy

    return np.stack((u, v), axis=-1)


def depth_2_point_cloud(depth_image, color_image, intrinsics_matrix, extrinsics_matrix, id):
    # # Convert depth image to Open3D format
    # depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

    # # Convert color image to Open3D format
    # color_o3d = o3d.geometry.Image(color_image.astype(np.uint8))

    # # Create RGBD image
    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #     color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=1000.0, convert_rgb_to_intensity=False
    # )

    # # Create camera intrinsics
    # fx = intrinsics_matrix[0, 0]
    # fy = intrinsics_matrix[1, 1]
    # cx = intrinsics_matrix[0, 2]
    # cy = intrinsics_matrix[1, 2]
    # width = depth_image.shape[1]
    # height = depth_image.shape[0]

    # intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # # Create point cloud from RGBD image
    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics, extrinsics_matrix)

    # # rot_cam_to_world = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    # # T_cam = np.eye(4)

    # # T_cam[:3, :3] = rot_cam_to_world @ extrinsics_matrix[:3, :3]
    # # T_cam[:3, 3] = extrinsics_matrix[:3, 3]

    # # # Transform the point cloud using the camera extrinsics
    # # # E_inv = np.linalg.inv(T_cam)
    # # pcd.transform(extrinsics_matrix)

    # return pcd

    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    cx = intrinsics_matrix[0, 2]
    cy = intrinsics_matrix[1, 2]

    # Get image dimensions
    height, width = depth_image.shape

    # Create a meshgrid for pixel coordinates (i, j)
    i, j = np.indices((height, width))

    # Unproject the depth map to 3D space in camera coordinates
    z = depth_image
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy

    # Stack x, y, z to form the point cloud in camera space (shape: (N, 3))
    point_cloud_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Apply extrinsics (transform from camera space to world space)
    # Add a 1 to each point for homogeneous coordinates (shape: (N, 4))
    ones = np.ones((point_cloud_camera.shape[0], 1))
    point_cloud_camera_homogeneous = np.hstack((point_cloud_camera, ones))

    # Apply the extrinsics matrix
    # extrinsics_matrix[:3, 3] = 0  # remove translation
    # point_cloud_world_homogeneous = (np.linalg.inv(extrinsics_matrix) @ point_cloud_camera_homogeneous.T).T
    point_cloud_world_homogeneous = np.linalg.inv(extrinsics_matrix) @ point_cloud_camera_homogeneous.T

    point_cloud_world_homogeneous = point_cloud_camera_homogeneous

    # Drop the homogeneous coordinate to get the final point cloud in world space
    point_cloud_world = point_cloud_world_homogeneous[:, :3]

    # Reshape the color image to align with the point cloud (shape: (N, 3))
    colors = color_image.reshape(-1, 3) / 255.0  # Normalize RGB to [0, 1]
    point_cloud_with_color = np.hstack((point_cloud_world, colors))
    # import pdb

    # pdb.set_trace()
    output_file = "/home/lucky/translate" + str(id) + ".csv"
    np.savetxt(output_file, point_cloud_with_color, delimiter=",", fmt="%.6f", header="X,Y,Z,R,G,B")

    # Return the point cloud and the corresponding colors
    return point_cloud_world, colors


def visualize_reprojected_depth(depth_map, rgb_image, points_2d, rgb_values, img_shape):
    """
    Visualizes the reprojected depth map and RGB image.
    """
    # Create empty images for depth and color
    reprojected_rgb = np.zeros((2048, 3060, 3), dtype=np.uint8)
    reprojected_depth = np.zeros((2048, 3060), dtype=np.float32)

    # Round to nearest integer pixel values
    points_2d = np.round(points_2d).astype(int)

    # Filter points that are within the image bounds
    valid_indices = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < 2048) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < 3060)
    )

    # Fill in the reprojected images
    reprojected_rgb[points_2d[valid_indices, 1], points_2d[valid_indices, 0]] = rgb_values[valid_indices]
    reprojected_depth[points_2d[valid_indices, 1], points_2d[valid_indices, 0]] = depth_map.reshape(-1)[valid_indices]

    # Visualize the reprojected depth map and RGB image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(reprojected_rgb)
    plt.title("Reprojected RGB Image")

    plt.subplot(1, 2, 2)
    plt.imshow(reprojected_depth, cmap="viridis")
    plt.colorbar(label="Depth (units)")
    plt.title("Reprojected Depth Map")

    plt.show()


if __name__ == "__main__":

    source_suffix = "8"
    target_suffix = "18"

    # path = IMG_OUT_PATH / Path("10-10-15-52_Dora_Turn-it-into-an-anime_42_5.0_0.5_2.0_0.2/30000/")
    path = IMG_OUT_PATH / Path("10-11-17-17_Dora_as-if-it-were-by-modigliani_42_5.0_0.5_2.0_0.2/30000/")
    source_depth_path = path / str(source_suffix + "_depth.pt")
    source_image_path = path / str(source_suffix + "_render.png")

    target_depth_path = path / str(target_suffix + "_depth.pt")
    target_image_path = path / str(target_suffix + "_render.png")

    transform_path = Path("/home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Dora_grn/transforms.json")
    transform_data = load_transform_json(transform_path)

    camera_frames = transform_data["frames"]

    size0, intrinsics0, source_transform = get_frame_data(camera_frames, source_suffix)
    size1, intrinsics1, target_transform = get_frame_data(camera_frames, target_suffix)

    assert np.array_equal(intrinsics0, intrinsics1)
    K = intrinsics0
    K[:2, :] /= 4

    D_source = torch.load(source_depth_path, weights_only=True).detach().cpu()
    maxima = torch.max(D_source)
    mask = D_source == maxima
    D_source[mask] = 0

    D_target = torch.load(target_depth_path, weights_only=True).detach().cpu()
    maxima = torch.max(D_target)
    mask = D_target == maxima
    D_target[mask] = 0

    source_color_image = Image.open(source_image_path)
    source_color_image = np.asarray(source_color_image)

    target_color_image = Image.open(target_image_path)
    target_color_image = np.asarray(target_color_image)

    # Example usage
    # Camera intrinsics (fx, fy, cx, cy) for two cameras

    # Camera extrinsics (rotation + translation matrices)
    extrinsics_cam1 = source_transform
    extrinsics_cam2 = target_transform

    # Depth map and RGB image from the first camera (example data)
    source_depth_map = D_source.numpy()
    source_rgb_image = source_color_image

    # Depth map and RGB image from the second camera (example data)
    target_depth_map = D_target.numpy()
    target_rgb_image = target_color_image

    pcd_source = depth_2_point_cloud(
        depth_image=source_depth_map,
        color_image=source_color_image,
        intrinsics_matrix=K,
        extrinsics_matrix=source_transform,
        id=source_suffix,
    )

    pcd_target = depth_2_point_cloud(
        depth_image=target_depth_map,
        color_image=target_color_image,
        intrinsics_matrix=K,
        extrinsics_matrix=target_transform,
        id=target_suffix,
    )

    # Create a rotation matrix for 180 degrees around the x-axis
    # rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # rotation_matrix_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    # pcd_rotated = copy.deepcopy(pcd_original)
    # pcd_rotated = pcd_rotated.rotate(rotation_matrix, center=(0, 0, 0))

    # pcd_rotated_y = copy.deepcopy(pcd_rotated)
    # pcd_rotated_y.rotate(rotation_matrix_y, center=(0, 0, 0))

    # Visualize both the original and rotated point clouds
    # o3d.visualization.draw_geometries([pcd_source, pcd_target])

    # # Unproject to 3D
    # points_3d, rgb_values = unproject_to_3d(depth_map, rgb_image, intrinsics_cam1)

    # visualize_point_cloud(points_3d, rgb_values)

    # # Transform to second camera's coordinate system
    # points_3d_transformed = transform_points(points_3d, extrinsics_cam1, extrinsics_cam2)

    # # Reproject to 2D for the second camera
    # points_2d = reproject_to_2d(points_3d_transformed, intrinsics_cam2)

    # # Visualize the reprojected depth and RGB image
    # visualize_reprojected_depth(depth_map, rgb_image, points_2d, rgb_values, (480, 640))
