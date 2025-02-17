import numpy as np
import matplotlib.pyplot as plt
import plyfile
from igs2gs.igs2gs_metrics.data_util import load_camera_data, parse_intrinsic, parse_transform

from pathlib import Path


def load_ply(file_path):
    # Load the PLY file
    ply_data = plyfile.PlyData.read(file_path)

    # Extract the vertices
    vertices = np.vstack([ply_data["vertex"]["x"], ply_data["vertex"]["y"], ply_data["vertex"]["z"]]).T

    return vertices


def visualize_depth(depth_map):

    # Plotting the depth map
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap="viridis")
    plt.colorbar(label="Depth (units)")
    plt.title("Depth Map Visualization")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.show()


if __name__ == "__main__":
    point_cloud_file_path = "/home/lucky/eick/splat.ply"  # Replace with your PLY file path
    point_cloud = load_ply(point_cloud_file_path)

    camera_json = Path(
        "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/10-16-18-18_Dora_as-if-it-were-by-modigliani_42_5.0_0.5_2.0_0.2/30000/camera_8.json"
    )
    camera_data = load_camera_data(camera_json)
    intrinsics = parse_intrinsic(camera_data)

    extrinsics = parse_transform(camera_data["camera_to_world"])

    point_cloud_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Apply the extrinsic matrix to transform the point cloud to camera coordinates
    camera_coords = np.linalg.inv(extrinsics) @ point_cloud_homogeneous.T
    depths = camera_coords[2, :]

    # Apply the intrinsic matrix to project the point to image coordinates
    image_coords_homogeneous = intrinsics @ camera_coords[:3]

    # Normalize the coordinates to get the 2D image coordinates
    uv_coord = image_coords_homogeneous[:2] / image_coords_homogeneous[2]

    # Mirror the x-axis to match the image coordinates
    uv_coord[0] = intrinsics[0, 2] * 2 - uv_coord[0]

    # Clip the coordinates to the image size
    # uv_coord = np.clip(uv_coord, 0, np.array([intrinsics[0, 2] * 2, intrinsics[1, 2] * 2]) - 1)

    # uv_coord = np.clip(uv_coord, 0, np.array([intrinsics[0, 2] * 4, intrinsics[1, 2] * 4]) - 1)
    # Visualize the depth map

    image = np.zeros((np.int64(intrinsics[1, 2] * 2), np.int64(intrinsics[0, 2] * 2)))

    # check if the uv_coord is within the image size
    valid = np.logical_and(uv_coord[0] >= 0, uv_coord[0] < image.shape[1])
    valid = np.logical_and(valid, uv_coord[1] >= 0)
    valid = np.logical_and(valid, uv_coord[1] < image.shape[0])

    uv_coord = uv_coord[:, valid]
    depths = depths[valid]

    image[uv_coord[1].astype(int), uv_coord[0].astype(int)] = depths * -1

    visualize_depth(image)
