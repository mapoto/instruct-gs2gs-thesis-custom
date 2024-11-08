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
from clip_metrics_batch import ClipSimilarity

PHOTODOME_GRID = np.array(
    [[18, 17, 9, 2, 33, 32, 24], [19, 11, 10, 3, 35, 34, 26], [20, 13, 12, 5, 4, 36, 28], [22, 15, 14, 7, 6, 37, 30]]
)
PHOTODOME_LINE = [1, 8, 16, 21, 23, 25, 27, 29, 31]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def pixel_L1_loss(masked_batch: torch.Tensor, intersection_batch: torch.Tensor, masks: torch.Tensor):
    # Scale the batches to the range [0, 255]
    masked_batch = (masked_batch + 1) * 255
    intersection_batch = (intersection_batch + 1) * 255

    # Expand the masks to match the number of channels in the batches
    masks = masks.unsqueeze(1).expand(-1, masked_batch.size(1), -1, -1)

    filtered_batch1 = masked_batch * masks
    filtered_batch2 = intersection_batch * masks

    # Compute the absolute differences
    abs_diff = torch.abs(filtered_batch1 - filtered_batch2)

    # Compute the mean of the absolute differences
    mean_abs_diff = torch.mean(abs_diff)

    return mean_abs_diff.item()


def get_largest_patch_with_min_holes(binary_mask: np.ndarray):

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=4)

    # Find the largest connected component (excluding the background)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Get the bounding box of the largest connected component
    x, y, w, h = (
        stats[largest_component, cv2.CC_STAT_LEFT],
        stats[largest_component, cv2.CC_STAT_TOP],
        stats[largest_component, cv2.CC_STAT_WIDTH],
        stats[largest_component, cv2.CC_STAT_HEIGHT],
    )

    # Crop the image using the bounding box
    largest_patch = binary_mask[y : y + h, x : x + w]

    a = y
    b = y + h
    c = x
    d = x + w
    return a, b, c, d


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


def compute_clip(
    masked_batch: list,
    intersection_batch: list,
    masks_batch: list,
    clip_model: ClipSimilarity,
):

    # masked_batch = masked_batch * masks_tensor
    # masked_batch = masked_batch.float().to(DEVICE)

    # intersection_batch = intersection_batch * masks_tensor
    # intersection_batch = intersection_batch.float().to(DEVICE)

    results = []
    for i in range(len(masked_batch)):
        img1 = masked_batch[i].float().to(DEVICE)
        img2 = intersection_batch[i].float().to(DEVICE)
        mask = masks_batch[i].float().to(DEVICE)

        val = clip_model.image_similarity(img1, img2, mask).item()

    # val = lpips_model(masked_batch, intersection_batch).item()

    return np.array(results).mean()


def compute_clip_per_iter(data_dir, adjacency_batch, lpips_model, save_folder, results_name="Ephra"):

    depths = load_all_depths(data_dir, len(adjacency_batch))
    renders = load_all_colors(data_dir, len(adjacency_batch))
    cameras = load_all_camera_data(data_dir, len(adjacency_batch))

    lpips_batch = np.eye(len(adjacency_batch))

    lpips_plane = np.zeros((4, 7))
    lpips_line = np.zeros((9))

    l1_plane = np.zeros((4, 7))
    l1_line = np.zeros((9))

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

            # a, b, c, d = get_largest_patch_with_min_holes(depth_intersection_binary.astype(np.uint8))

            # Color interesection image
            color_intersection = color_intersection * 255 * depth_intersection_binary[:, :, None]
            color_intersection = color_intersection.astype(np.uint8)
            # color_intersection = color_intersection[a:b, c:d]
            # Add depth_intersection_binary as alpha channel to the color_intersection
            # color_intersection = np.concatenate([color_intersection, depth_intersection_binary[:, :, None]], axis=2)
            # color_intersection = get_largest_patch_with_min_holes(color_intersection)

            # Store color intersection as an image
            # cv2.imwrite(batch_folder / f"color_intersection_{adj_idx}.png", color_intersection)

            # applies mask to the color image of the target camera
            colors_masked = (
                central_render.reshape(depth_intersection_binary.shape[0], depth_intersection_binary.shape[1], 3)
                * 255
                * depth_intersection_binary[:, :, None]
            )
            colors_masked = colors_masked.astype(np.uint8)
            # BGR to RGB
            colors_masked = colors_masked[:, :, [2, 1, 0]]
            # colors_masked = colors_masked[a:b, c:d]
            # Add depth_intersection_binary as alpha channel to the colors_masked
            # colors_masked = np.concatenate([colors_masked, depth_intersection_binary[:, :, None]], axis=2)
            # colors_masked = get_largest_patch_with_min_holes(colors_masked)

            # Store masked color image
            # cv2.imwrite(batch_folder / f"colors_masked_{adj_idx}and{central_idx}.png", colors_masked)

            # Store depth intersection as an image
            # cv2.imwrite(
            #     batch_folder / f"depth_intersection_{adj_idx}to{central_idx}.png", depth_intersection_binary * 255
            # )

            color_intersection_tensor = torch.from_numpy(color_intersection).permute(2, 0, 1).unsqueeze(0).float()
            colors_masked_tensor = torch.from_numpy(colors_masked).permute(2, 0, 1).unsqueeze(0).float()

            stacked_intersection_tensor.append(color_intersection_tensor)
            stacked_masked_tensor.append(colors_masked_tensor)
            stacked_binary_tensor.append(torch.from_numpy(depth_intersection_binary).unsqueeze(0).float())
            pass

        # Compute LPIPS
        average_lpips = compute_clip(
            masked_batch=stacked_masked_tensor,
            intersection_batch=stacked_intersection_tensor,
            clip_model=ClipSimilarity(name="./ViT-L/14"),
            masks_batch=stacked_binary_tensor,
        )

        # Compute L1 loss
        # average_l1 = pixel_L1_loss(
        #     masked_batch=stacked_masked_tensor,
        #     intersection_batch=stacked_intersection_tensor,
        #     masks=stacked_binary_tensor,
        # )

        # print(f"Average L1 Loss: {average_l1}")

        adjusted_idx = central_idx + 1  # FRAMES ARE 1 INDEXED

        if results_name == "Ephra":
            if central_idx > 26:
                adjusted_idx = central_idx + 2  # MISSING CAMERA

        print(f"Adjusted Index: {adjusted_idx} <- {central_idx}")
        if adjusted_idx in PHOTODOME_LINE:
            x = find_position_in_line(adjusted_idx)
            lpips_line[x] = average_lpips
        else:
            x, y = find_position_in_grid(adjusted_idx)
            lpips_plane[x, y] = average_lpips

        pass

    if results_name == "Ephra":
        # Interpolate missing values
        lpips_plane[2, 6] = np.mean(
            [
                lpips_plane[1, 6],
                lpips_plane[3, 6],
                lpips_plane[2, 5],
                lpips_plane[1, 5],
                lpips_plane[3, 5],
                lpips_line[6],
                lpips_line[5],
                lpips_line[8],
            ]
        )

    return lpips_plane, lpips_line


def compute_rmse_plane(lpips_plane):
    return np.sqrt(np.mean(np.square(lpips_plane)))


def compute_rmse_line(lpips_line):
    return np.sqrt(np.mean(np.square(lpips_line)))


if __name__ == "__main__":

    root = sys.argv[1]

    # root = Path(
    #     "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/10-22-15-02_Ephra_turn-him-into-Tolkien-Elf_42_5.0_0.5_2.0_0.2"
    # )

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

    # root = Path(root)

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
            if (int(iter) % 100 == 0) or (int(iter) % 2500 == 1):
                folders.append(folder)

    folders.sort(reverse=True)

    print(len(folders))
    print(DEVICE)

    lpips_dict = {}

    save_folder = debug_root_folder / root.name
    save_folder.mkdir(parents=True, exist_ok=True)

    save_path = save_folder / "eval_lpips.csv"

    with open(save_path, "w") as f:
        f.write("iter,min_val,max_val,mean_val,std_val\n")

    for folder in folders:
        iter = folder.stem

        name = folder.parent.stem

        image_save_folder = save_folder / iter
        image_save_folder.mkdir(parents=True, exist_ok=True)

        print(f"Computing for iteration {folder.stem}")

        lpips_plane, lpips_line = compute_clip_per_iter(
            data_dir=folder,
            adjacency_batch=adjacency_batch,
            lpips_model=lpips_model,
            save_folder=image_save_folder,
        )

        lpips = np.concatenate([lpips_plane.flatten(), lpips_line.flatten()])

        min_val, max_val, mean_val, std_val = compute_stats(lpips)

        # save plane lpips
        np.savetxt(image_save_folder / "lpips_plane.csv", lpips_plane, delimiter=",")

        # save line lpips
        np.savetxt(image_save_folder / "lpips_line.csv", lpips_line, delimiter=",")

        with open(save_path, "a+") as f:
            f.write(f"{iter},{min_val},{max_val},{mean_val},{std_val}\n")
