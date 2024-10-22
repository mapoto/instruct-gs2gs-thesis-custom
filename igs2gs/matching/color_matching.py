import numpy as np
import matplotlib.pyplot as plt
import plyfile
from data_util import load_camera_data, parse_intrinsic, parse_transform
import torch
from pathlib import Path
import cv2
from PIL import Image

import lpips


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


def validate_image_size(uv_coord, width, height):

    # check if the uv_coord is within the image size
    valid = np.logical_and(uv_coord[0] >= 0, uv_coord[0] < width)
    valid = np.logical_and(valid, uv_coord[1] >= 0)
    valid = np.logical_and(valid, uv_coord[1] < height)

    uv_coord = uv_coord[:, valid]
    return uv_coord, valid


def compute_lips(source_color_image: torch.Tensor, target_color_image: torch.Tensor):

    # Downsample the image
    source_color_image = torch.nn.functional.interpolate(source_color_image, scale_factor=0.25, mode="bilinear")
    target_color_image = torch.nn.functional.interpolate(target_color_image, scale_factor=0.25, mode="bilinear")

    # Compute LPIPS
    loss_fn_vgg = lpips.LPIPS(net="vgg")
    loss_fn_alex = lpips.LPIPS(net="alex")

    # Compute the LPIPS
    loss_vgg = loss_fn_vgg(source_color_image, target_color_image)
    loss_alex = loss_fn_alex(source_color_image, target_color_image)

    return loss_vgg, loss_alex


if __name__ == "__main__":
    DATA_PATH = Path(
        "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/10-16-18-18_Dora_as-if-it-were-by-modigliani_42_5.0_0.5_2.0_0.2/30000"
    )

    # Load the point cloud from source camera
    source = "18"
    points18, colors18 = point_cloud_csv("/home/lucky/point_cloud_18_new.csv")

    # Target camera intrinsics and extrinsics
    target = "8"
    camera_json = DATA_PATH / f"camera_{target}.json"
    target_color_path = DATA_PATH / f"{target}_render.png"
    camera_data = load_camera_data(camera_json)

    intrinsics = parse_intrinsic(camera_data)
    extrinsics = parse_transform(camera_data["camera_to_world"])

    # Convert the point cloud to homogeneous coordinates source camera
    point_cloud_homogeneous = np.hstack((points18, np.ones((points18.shape[0], 1))))

    # Apply the extrinsic matrix to transform the point cloud to camera coordinates
    camera_coords = np.linalg.inv(extrinsics) @ point_cloud_homogeneous.T

    # Apply the intrinsic matrix to project the point to image coordinates
    image_coords_homogeneous = intrinsics @ camera_coords[:3]

    # Normalize the coordinates to get the 2D image coordinates
    uv_coord = image_coords_homogeneous[:2] / image_coords_homogeneous[2]

    # Mirror the x-axis to match the image coordinates
    uv_coord[0] = intrinsics[0, 2] * 2 - uv_coord[0]

    # Depth map from the target camera
    depth8 = read_depth(depth_path=DATA_PATH / f"{target}_depth.pt")
    depth8_uv = depth_as_normalized_camera_coords(depth8.numpy(), intrinsics)
    depths = depth8_uv[2, :]

    width = np.int64(intrinsics[0, 2] * 2)
    height = np.int64(intrinsics[1, 2] * 2)

    color_intersection = np.zeros(
        (
            height,
            width,
            3,
        )
    )
    depth_intersection = np.zeros((height, width))

    uv_coord, valid = validate_image_size(uv_coord, width, height)

    colors = colors18[valid]
    depths = depths[valid]

    # BGR to RGB
    rgb = colors[:, [2, 1, 0]]  # BGR to RGB
    # colors = colors[[2, 1, 0], :]  # BGR to RGB

    # Compute the depth intersection mask, If there is x,y duplicate takes the smaller one (most front)

    negate_depths = depths * -1
    depth_intersection[uv_coord[1].astype(int), uv_coord[0].astype(int)] = negate_depths

    # Compute the color image from point cloud in target camera perpective
    color_intersection[uv_coord[1].astype(int), uv_coord[0].astype(int)] = rgb

    # Covert to black and white image
    depth_intersection_binary = np.where(depth_intersection > 0, 1, 0)

    # Color interesection image
    color_intersection = color_intersection * 255 * depth_intersection_binary[:, :, None]
    color_intersection = color_intersection.astype(np.uint8)

    # Store color intersection as an image
    cv2.imwrite(f"/home/lucky/color_intersection_{source}.png", color_intersection)

    # applies mask to the color image of the target camera
    target_color_image = cv2.imread(target_color_path)
    colors_masked = target_color_image * depth_intersection_binary[:, :, None]

    # Store masked color image
    cv2.imwrite(f"/home/lucky/colors_masked_{target}.png", colors_masked)

    # Store depth intersection as an image
    cv2.imwrite(f"/home/lucky/depth_intersection_{source}to{target}.png", depth_intersection_binary * 255)

    color_intersection_tensor = torch.from_numpy(color_intersection / 255).permute(2, 0, 1).unsqueeze(0).float()
    colors_masked_tensor = torch.from_numpy(colors_masked / 255).permute(2, 0, 1).unsqueeze(0).float()

    # Compute LPIPS
    loss_vgg, loss_alex = compute_lips(color_intersection_tensor, colors_masked_tensor)

    print(f"VGG LPIPS: {loss_vgg.item()}")
    print(f"Alex LPIPS: {loss_alex.item()}")
