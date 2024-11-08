from pathlib import Path
import numpy as np
import torch
from PIL import Image
from igs2gs.igs2gs_metrics.data_util import *
import cv2


def unproject_depth_image(depth_image, intrinsic_matrix, extrinsic_matrix):
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

    # Convert to camera coordinates
    camera_coords = np.linalg.inv(intrinsic_matrix) @ uv_coord

    # Convert to homogeneous coordinates
    point_cloud_homogeneous = np.vstack((camera_coords, np.ones_like(i)))

    # Convert to world coordinates
    world_coords = extrinsic_matrix @ point_cloud_homogeneous

    # Extract x, y, z coordinates
    x = world_coords[0, :]
    y = world_coords[1, :]
    z = world_coords[2, :]
    point_cloud = np.vstack((x, y, z)).T

    return point_cloud


def save_point_cloud_to_csv(point_cloud, filename):
    np.savetxt(filename, point_cloud, delimiter=",", header="x,y,z", comments="")


def read_depth(depth_path):
    depth_map = torch.load(depth_path, weights_only=True).detach().cpu()
    maxima = torch.max(depth_map)
    mask = depth_map == maxima
    depth_map[mask] = 0
    return depth_map


IMG_OUT_PATH = Path("/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/")

if __name__ == "__main__":

    id = "4"

    path = IMG_OUT_PATH / Path(
        "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/10-22-15-02_Ephra_turn-him-into-Tolkien-Elf_42_5.0_0.5_2.0_0.2/35000/"
    )

    unprojects = Path("/home/lucky/unprojects")
    unprojects.mkdir(exist_ok=True)

    for id in range(36):
        id = str(id)
        depth_path = path / str(id + "_depth.pt")
        image_path = path / str(id + "_render.png")
        camera_path = path / f"camera_{id}.json"

        source_camera_data = load_camera_data(camera_path)

        K = parse_intrinsic(source_camera_data)
        E = parse_transform(source_camera_data["camera_to_world"])

        depth = read_depth(depth_path).numpy()
        color = read_colors_from_image(image_path)

        # Example usage

        point_cloud = unproject_depth_image(depth, K, E)
        point_cloud_with_color = np.hstack((point_cloud, color))

        save_point_cloud_to_csv(point_cloud_with_color, unprojects / f"ephra_point_cloud_{id}_new.csv")

        # normalize depth
        norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        norm = norm * 255
        norm = norm.astype(np.uint8)
        # save depth as image
        cv2.imwrite(unprojects / f"ephra_depth_{id}.png", norm)
