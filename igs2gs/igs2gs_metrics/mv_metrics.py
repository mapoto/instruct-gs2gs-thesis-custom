import numpy as np
import matplotlib.pyplot as plt
import plyfile
import torch
from pathlib import Path
import cv2
from PIL import Image

import unproject_depth
import data_util
import pandas as pd
from unproject_depth import unproject_depth_image, save_point_cloud_to_csv
from color_matching import compute_lips, compute_intersection
import copy
import json
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import sys
import colour

PHOTODOME_GRID = np.array(
    [[18, 17, 9, 2, 33, 32, 24], [19, 11, 10, 3, 35, 34, 26], [20, 13, 12, 5, 4, 36, 28], [22, 15, 14, 7, 6, 37, 30]]
)
PHOTODOME_LINE = [1, 8, 16, 21, 23, 25, 27, 29, 31]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_rmse_filter(
    masked_batch: list,
    intersection_batch: list,
    masks_batch: list,
):

    results = []
    for i in range(len(masked_batch)):
        img1 = masked_batch[i].float().to(DEVICE)
        img2 = intersection_batch[i].float().to(DEVICE)
        mask = masks_batch[i].float().to(DEVICE)
        mask = mask.unsqueeze(1)  # Now expanded_mask has shape [1, 1, 765, 512]
        mask = mask.repeat(1, 3, 1, 1)  # Now expanded_mask has shape [1, 1, 765, 512]

        # Mask img1 so that it only contains the value of 1 in mask
        img1 = img1[mask == 1] + 1 * 255
        img2 = img2[mask == 1] + 1 * 255

        abs_diff = torch.abs(img1 - img2)

        squared_diff = abs_diff**2
        mean_squared_diff = torch.mean(squared_diff)

        rmse = torch.sqrt(mean_squared_diff).item()

        torch.cuda.empty_cache()

    return rmse


def find_position_in_grid(idx):
    for i in range(PHOTODOME_GRID.shape[0]):
        for j in range(PHOTODOME_GRID.shape[1]):
            if PHOTODOME_GRID[i, j] == idx:
                return i, j
    return -1, -1


def find_position_in_line(idx):
    for i in range(len(PHOTODOME_LINE)):
        if PHOTODOME_LINE[i] == idx:
            return i
    return -1


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


def compute_stats(average_lpips: np.ndarray):

    min_val = np.min(average_lpips)
    max_val = np.max(average_lpips)
    mean_val = np.mean(average_lpips)
    std_val = np.std(average_lpips)

    return (
        min_val,
        max_val,
        mean_val,
        std_val,
    )


def compute_lpips_torch(
    masked_batch: list,
    intersection_batch: list,
    masks_batch: list,
    lpips_model: LearnedPerceptualImagePatchSimilarity,
):

    results = []
    for i in range(len(masked_batch)):
        img1 = masked_batch[i].float().to(DEVICE)
        img2 = intersection_batch[i].float().to(DEVICE)
        mask = masks_batch[i].float().to(DEVICE)

        lpips_val = lpips_model(img1, img2)

        # Unnecessary to include mask here since it is already applied to the images (just to correspond to original implementation)
        pips = (lpips_val * mask).flatten(1).sum(-1)
        pips = pips / mask.flatten(1).sum(-1)

        results.append(pips.mean().item())
        torch.cuda.empty_cache()

    # val = lpips_model(masked_batch, intersection_batch).item()

    return np.array(results).mean()


def compute_ciede2000(
    masked_batch: list,
    intersection_batch: list,
    masks_batch: list,
):

    results = []
    for i in range(len(masked_batch)):
        img1 = masked_batch[i].float().squeeze(0)
        img2 = intersection_batch[i].float().squeeze(0)
        mask = masks_batch[i].float()

        mask = mask.repeat(3, 1, 1)  # Now expanded_mask has shape [1, 1, 765, 512]

        # Convert RGB to CIELAB
        img1 = img1.permute(1, 2, 0).cpu().numpy()
        img2 = img2.permute(1, 2, 0).cpu().numpy()
        mask = mask.permute(1, 2, 0).cpu().numpy()

        img1 = cv2.cvtColor(img1.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        img2 = cv2.cvtColor(img2.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)

        # Mask img1 so that it only contains the value of 1 in mask

        img1 = img1[mask == 1] + 1 * 255
        img2 = img2[mask == 1] + 1 * 255

        img1 = img1.reshape(-1, 3)
        img2 = img2.reshape(-1, 3)

        delta_E = colour.delta_E(img1, img2)

        mean_delta_E = np.mean(delta_E)

        results.append(mean_delta_E)
        torch.cuda.empty_cache()

    return np.array(results).mean()


def compute_metrics(data_dir, adjacency_batch, lpips_model, save_folder, results_name="Ephra"):

    depths = load_all_depths(data_dir, len(adjacency_batch))
    renders = load_all_colors(data_dir, len(adjacency_batch))
    cameras = load_all_camera_data(data_dir, len(adjacency_batch))

    remse = []
    lpips = []
    ciede = []

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

        stacked_intersection_tensor = []
        stacked_masked_tensor = []
        stacked_binary_tensor = []

        print(f"Computing for camera {central_idx}")

        # For every adjacent camera in the batch
        for adj_idx in list(adjacents_idx):
            if adj_idx == central_idx:
                continue

            # print("Computing pair for ", central_idx, " and ", adj_idx)

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

            color_intersection_tensor = (
                torch.from_numpy((color_intersection / 255) - 1).permute(2, 0, 1).unsqueeze(0).float()
            )
            colors_masked_tensor = torch.from_numpy((colors_masked / 255) - 1).permute(2, 0, 1).unsqueeze(0).float()

            stacked_intersection_tensor.append(color_intersection_tensor)
            stacked_masked_tensor.append(colors_masked_tensor)
            stacked_binary_tensor.append(torch.from_numpy(depth_intersection_binary).unsqueeze(0).float())
            pass

        # Compute LPIPS
        camera_lpips = compute_lpips_torch(
            masked_batch=stacked_masked_tensor,
            intersection_batch=stacked_intersection_tensor,
            lpips_model=lpips_model,
            masks_batch=stacked_binary_tensor,
        )

        camera_rmse = compute_rmse_filter(
            masked_batch=stacked_masked_tensor,
            intersection_batch=stacked_intersection_tensor,
            masks_batch=stacked_binary_tensor,
        )
        camera_ciede2000 = compute_ciede2000(
            masked_batch=stacked_masked_tensor,
            intersection_batch=stacked_intersection_tensor,
            masks_batch=stacked_binary_tensor,
        )

        print(f"Average LPIPS for camera {central_idx}: {camera_lpips}")
        print(f"RMSE Filter for camera {central_idx}: {camera_rmse}")

        remse.append(camera_rmse)
        lpips.append(camera_lpips)
        ciede.append(camera_ciede2000)

    return lpips, remse, ciede


def compute_rmse_plane(lpips_plane):
    return np.sqrt(np.mean(np.square(lpips_plane)))


def compute_rmse_line(lpips_line):
    return np.sqrt(np.mean(np.square(lpips_line)))


if __name__ == "__main__":

    root = sys.argv[1]
    # root = "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs_low/10-22-15-02_Ephra_turn-him-into-Tolkien-Elf_42_5.0_0.5_2.0_0.2/"
    name = "Ephra"
    types = ["Simon", "Dora", "Ephra", "Irene"]
    for t in types:
        if t in root:
            name = t
            break
    if name == "":
        print("Invalid name")
        sys.exit(1)

    print(name)
    root = Path(root)

    debug_root_folder = Path("/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/")

    # Load adjacency batches
    cam_adj_matrices = {
        "Simon": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/simon.csv")).values),
        "Dora": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/dora.csv")).values),
        "Ephra": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/ephra.csv")).values),
        "Irene": torch.tensor(pd.read_csv(Path("./igs2gs/adj_matrices/irene.csv")).values),
    }

    adjacency_batch = cam_adj_matrices[name].numpy()
    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(DEVICE)

    # Iterate all folders in root
    folders = []
    for folder in root.iterdir():
        if folder.is_dir():
            iter = folder.stem
            if int(iter) == 35000 or (int(iter) % 2500 == 1) or int(iter) == 30000:
                folders.append(folder)

    folders.sort(reverse=True)

    print(len(folders))
    print(DEVICE)

    save_folder = debug_root_folder / root.name
    save_folder.mkdir(parents=True, exist_ok=True)

    for folder in folders:
        iter = folder.stem

        name = folder.parent.stem
        image_save_folder = save_folder / iter

        print(f"Computing for iteration {folder.stem}")

        lpips, rmse, ciede = compute_metrics(
            data_dir=folder,
            adjacency_batch=adjacency_batch,
            lpips_model=lpips_model,
            save_folder=image_save_folder,
        )

        # save plane rmse
        np.savetxt(save_folder / f"{iter}_ciede2000.csv", ciede, delimiter=",")
        np.savetxt(save_folder / f"{iter}_rmse.csv", rmse, delimiter=",")
        np.savetxt(save_folder / f"{iter}_lpips.csv", lpips, delimiter=",")