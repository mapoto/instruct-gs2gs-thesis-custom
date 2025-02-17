import numpy as np
import matplotlib.pyplot as plt
import plyfile
import torch
from pathlib import Path
import cv2
from PIL import Image

import lpips
import unproject_depth
import data_util
import pandas as pd
from unproject_depth import unproject_depth_image, save_point_cloud_to_csv
from color_matching import compute_lips, compute_intersection
import copy
import json


def load_all_depths(root: Path, num_cameras: int):
    depths = {}
    for i in range(num_cameras):
        depth_path = root / f"{i}_depth.pt"
        depths[i] = data_util.read_depth(depth_path)
    return depths


def load_all_colors(root: Path, num_cameras: int):
    colors = {}
    for i in range(num_cameras):
        color_path = root / f"{i}_render.png"
        color = data_util.read_colors_from_image(color_path)
        colors[i] = color
    return colors


def load_all_camera_data(root: Path, num_cameras: int):
    camera_data = {}
    for i in range(num_cameras):
        camera_path = root / f"camera_{i}.json"
        data = data_util.load_camera_data(camera_path)

        intrinsics = data_util.parse_intrinsic(data)
        extrinsics = data_util.parse_transform(data["camera_to_world"])

        camera_data[i] = {"intrinsics": intrinsics, "extrinsics": extrinsics}

    return camera_data


def compute_lpips_per_iter(data_dir, adjacency_batch, loss_fn_vgg, loss_fn_alex, save_folder):

    depths = load_all_depths(data_dir, len(adjacency_batch))
    renders = load_all_colors(data_dir, len(adjacency_batch))
    cameras = load_all_camera_data(data_dir, len(adjacency_batch))

    lpips_batch = np.eye(len(adjacency_batch))
    # For every central camera in the batch
    for central_idx in range(0, len(adjacency_batch)):
        adjacents_idx = np.where(adjacency_batch[central_idx] == 1)
        adjacents_idx = adjacents_idx[0]

        # Load the central camera data
        central_camera_data = cameras[central_idx]
        central_intrinsics = central_camera_data["intrinsics"]
        central_extrinsics = central_camera_data["extrinsics"]

        central_depth = depths[central_idx].numpy()
        central_render = renders[central_idx]

        batch_folder = save_folder / str(central_idx)
        batch_folder.mkdir(parents=True, exist_ok=True)

        # For every adjacent camera in the batch
        for adj_idx in list(adjacents_idx):
            if adj_idx == central_idx:
                continue

            print("Computing pair for ", central_idx, " and ", adj_idx)

            adj_camera_data = cameras[adj_idx]

            adj_K = adj_camera_data["intrinsics"]
            adj_E = adj_camera_data["extrinsics"]

            # Load the depth image of the adjacent camera
            adj_depth = depths[adj_idx].numpy()
            adj_color = renders[adj_idx]

            # Unproject the depth image of the adjacent camera
            adj_pc = unproject_depth_image(depth_image=adj_depth, intrinsic_matrix=adj_K, extrinsic_matrix=adj_E)

            # Compute intersection
            depth_intersection, color_intersection = compute_intersection(
                point_cloud=adj_pc,
                source_color=adj_color,
                target_intrinsics=central_intrinsics,
                target_extrinsics=central_extrinsics,
                target_depth=central_depth,
            )

            # Covert to black and white image
            depth_intersection_binary = np.where(depth_intersection > 0, 1, 0)

            # Color interesection image
            color_intersection = color_intersection * 255 * depth_intersection_binary[:, :, None]
            color_intersection = color_intersection.astype(np.uint8)

            # Store color intersection as an image
            cv2.imwrite(batch_folder / f"color_intersection_{adj_idx}.png", color_intersection)

            # applies mask to the color image of the target camera
            colors_masked = (
                central_render.reshape(depth_intersection_binary.shape[0], depth_intersection_binary.shape[1], 3)
                * 255
                * depth_intersection_binary[:, :, None]
            )
            colors_masked = colors_masked.astype(np.uint8)
            # BGR to RGB
            colors_masked = colors_masked[:, :, [2, 1, 0]]

            # Store masked color image
            cv2.imwrite(batch_folder / f"colors_masked_{adj_idx}and{central_idx}.png", colors_masked)

            # Store depth intersection as an image
            cv2.imwrite(
                batch_folder / f"depth_intersection_{adj_idx}to{central_idx}.png", depth_intersection_binary * 255
            )

            color_intersection_tensor = torch.from_numpy(color_intersection / 255).permute(2, 0, 1).unsqueeze(0).float()
            colors_masked_tensor = torch.from_numpy(colors_masked / 255).permute(2, 0, 1).unsqueeze(0).float()

            # Compute LPIPS
            loss_vgg, _ = compute_lips(
                source_color_image=color_intersection_tensor,
                target_color_image=colors_masked_tensor,
                use_vgg=True,
                loss_fn_vgg=loss_fn_vgg,
            )

            lpips_batch[int(central_idx)][int(adj_idx)] = loss_vgg.item()
            print(f"VGG LPIPS {str(central_idx)} to {str(adj_idx)}: {lpips_batch[central_idx][adj_idx]}")

            pass
        pass

    return lpips_batch


if __name__ == "__main__":

    debug_root_folder = Path("/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/")
    name = "Ephra"
    # Load adjacency batches

    cam_adj_matrices = {
        "Simon": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/simon.csv")).values),
        "Dora": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/dora.csv")).values),
        "Ephra": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/ephra.csv")).values),
        "Irene": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/irene.csv")).values),
    }

    adjacency_batch = cam_adj_matrices[name].numpy()

    loss_fn_vgg = lpips.LPIPS(net="vgg")
    loss_fn_alex = lpips.LPIPS(net="alex")

    root = Path(
        "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/10-22-15-02_Ephra_turn-him-into-Tolkien-Elf_42_5.0_0.5_2.0_0.2/"
    )

    # Iterate all folders in root
    folders = []
    for folder in root.iterdir():
        if folder.is_dir():
            iter = folder.stem
            if (int(iter) % 100 == 0) or (int(iter) % 2500 == 1):
                folders.append(folder)

    folders.sort(reverse=True)

    print(len(folders))

    lpips_dict = {}
    for folder in folders:
        iter = folder.stem

        name = folder.parent.stem

        image_save_folder = debug_root_folder / name / iter
        image_save_folder.mkdir(parents=True, exist_ok=True)

        print(f"Computing for iteration {folder.stem}")

        lpips_value = compute_lpips_per_iter(
            data_dir=folder,
            adjacency_batch=adjacency_batch,
            loss_fn_vgg=loss_fn_vgg,
            loss_fn_alex=loss_fn_alex,
            save_folder=image_save_folder,
        )

        lpips_dict[str(iter)] = lpips_value.tolist()
        save_path = debug_root_folder / name / f"lpips_{folder.stem}.csv"
        print(f"Saving to {save_path}")
        # Save as csv
        np.savetxt(save_path, lpips_value.tolist(), delimiter=",")
