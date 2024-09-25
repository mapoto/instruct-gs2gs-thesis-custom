import numpy as np
import torch
from plyfile import PlyData
import json
import tkinter as tk
from tkinter import filedialog
import gsplat
import sys
import matplotlib.pyplot as plt


def parse_ply(file_path):
    plydata = PlyData.read(file_path)
    vertices = plydata["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    scale = np.vstack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).T
    rotation = np.vstack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]).T
    print("Printing first 10 xyz values", xyz[:10])
    return xyz, scale, rotation


def load_transform_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def normalize_quaternions(quats):
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    normalized_quats = quats / norms
    return normalized_quats


def progress_update(current, total):
    progress = (current / total) * 100
    sys.stdout.write(f"\rProgress: {progress:.2f}%")
    sys.stdout.flush()


def project_and_plot_frame(means3d, scales, quats, frame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_matrix_viewmats = torch.tensor(frame["transform_matrix"], dtype=torch.float32).to(device)

    R = transform_matrix_viewmats[:3, :3]  # 3 x 3
    T = transform_matrix_viewmats[:3, 3:4]  # 3 x 1

    R_edit = torch.diag(torch.tensor([1, -1, -1], device=device, dtype=R.dtype))
    R = R @ R_edit

    R_inv = R.T
    T_inv = -R_inv @ T

    viewmats = torch.eye(4, device=R.device, dtype=R.dtype)
    viewmats[:3, :3] = R_inv
    viewmats[:3, 3:4] = T_inv

    fx = frame["fl_x"]
    fy = frame["fl_y"]
    cx = frame["cx"]
    cy = frame["cy"]
    img_height = frame["h"]
    img_width = frame["w"]
    glob_scale = 1.0
    block_width = 16
    clip_thresh = 0.01

    if torch.cuda.is_available():
        means3d = means3d.cuda()
        scales = scales.cuda()
        quats = quats.cuda()

    xys, depths, radii, conics, compensation, num_tiles_hit, cov3d = gsplat.project_gaussians(
        means3d.contiguous(),
        scales.contiguous(),
        glob_scale,
        quats.contiguous(),
        viewmats,
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        block_width,
        clip_thresh,
    )

    print("Checking projection results...")
    print(f"xys (2D projections):\n{xys[:10]}")
    print(f"depths (z-depths):\n{depths[:10]}")

    """Filter out invalid projections"""
    valid_indices = (
        (xys[:, 0] > 0) & (xys[:, 1] > 0) & (xys[:, 0] < img_width) & (xys[:, 1] < img_height) & (depths > clip_thresh)
    )
    xys_filtered = xys[valid_indices].cpu().numpy()

    print(f"Filtered xys (2D projections):\n{xys_filtered[:10]}")

    if xys_filtered.size == 0:
        print("No valid points to plot for this frame.")
        return

    """Plotting in 2D"""
    print("\nPlotting valid 2D projections...")
    plt.figure(figsize=(10, 8))
    plt.scatter(xys_filtered[:, 0], xys_filtered[:, 1], c="blue", s=1)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title(f'Valid 2D Projections for {frame["file_path"]}')
    plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
    plt.show()


def main():
    # File selection dialog
    root = tk.Tk()
    root.withdraw()

    ply_file_path = filedialog.askopenfilename(title="Select PLY File", filetypes=[("PLY files", "*.ply")])
    transform_json_path = filedialog.askopenfilename(
        title="Select Transform JSON File", filetypes=[("JSON files", "*.json")]
    )

    if not ply_file_path or not transform_json_path:
        print("File selection cancelled.")
        return

    print("Parsing PLY file...")
    xyz, scale, rotation = parse_ply(ply_file_path)
    print(f"Extracted {xyz.shape[0]} points from PLY file.")

    print("Loading transform JSON...")
    transform_data = load_transform_json(transform_json_path)

    means3d = torch.tensor(xyz, dtype=torch.float32)
    scales = torch.tensor(scale, dtype=torch.float32)

    print("Normalizing quaternions...")
    normalized_quats = normalize_quaternions(rotation)
    quats = torch.tensor(normalized_quats, dtype=torch.float32)

    if torch.cuda.is_available():
        print("CUDA is available. Moving tensors to GPU.")
        means3d = means3d.cuda()
        scales = scales.cuda()
        quats = quats.cuda()

    for frame in transform_data["frames"]:
        print(f"Processing frame: {frame['file_path']}")
        project_and_plot_frame(means3d, scales, quats, frame)


if __name__ == "__main__":
    main()
