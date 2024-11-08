import open3d as o3d
import os
import copy
import numpy as np
import json
import torch
from PIL import Image
from pathlib import Path
from igs2gs.igs2gs_metrics.data_util import get_frame_data, load_transform_json
import matplotlib.pyplot as plt


def main():

    IMG_OUT_PATH = Path("/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/")

    path = IMG_OUT_PATH / Path("09-26-15-49_Dora_Turn-it-into-an-anime_42_5.0_0.5_2.0_0.2/30000")
    depth_path = path / "8_depth.pt"
    D_source = torch.load(depth_path, weights_only=True).detach().cpu()
    maxima = torch.max(D_source)
    mask = D_source == maxima
    D_source[mask] = 0

    transform_path = Path("data/Dora_grn/transforms.json")
    transform_data = load_transform_json(transform_path)

    camera_frames = transform_data["frames"]
    source_size, source_intrinsic, source_transform = get_frame_data(camera_frames, "8")
    size1, intrinsics1, target_transform = get_frame_data(camera_frames, "8")
    source_extrinsic = torch.from_numpy(source_transform).float()
    # source_intrinsic = torch.from_numpy(intrinsics0).float()
    source_intrinsic[:, :2] = source_intrinsic[:, :2] / 4

    o3d_depth = o3d.geometry.Image(D_source.numpy())
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=int(source_size[0] / 4), height=int(source_size[1] / 4), intrinsic_matrix=source_intrinsic
    )

    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, intrinsic=o3d_intrinsics)

    o3d.visualization.draw([pcd])
    pass


if __name__ == "__main__":
    main()
