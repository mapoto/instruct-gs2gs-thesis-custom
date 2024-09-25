import re
import numpy as np
import os

from pathlib import Path

image_root_path = Path("/home/lucky/dataset/thesis_gauss_participants/Dora_grn/")
camera_names = [image_path.name for image_path in image_root_path.glob("*.[jpJP][npNP]*[gG$]")]


# Parse camera names
def parse_camera_name(name):
    match = re.match(r"(\d+)-([1-3BC])-([1-6])-([1-2])-\d+-\d+\.JPG", name)
    if match:
        session, column, bar, position = match.groups()
        column = {"1": 1, "2": 2, "3": 3, "B": 4, "C": 5}[column]
        return (int(session), column, int(bar), int(position))
    return None


# Create a list of parsed camera details
cameras = [parse_camera_name(name) for name in camera_names]

# Create adjacency matrix
num_cameras = len(cameras)
adj_matrix = np.zeros((num_cameras, num_cameras), dtype=int)

# Determine adjacency
for i, cam1 in enumerate(cameras):
    for j, cam2 in enumerate(cameras):
        if i != j:
            # Check if cameras are adjacent
            if (
                (cam1[1] == cam2[1] and cam1[2] == cam2[2] and abs(cam1[3] - cam2[3]) <= 1)
                or (cam1[1] == cam2[1] and abs(cam1[2] - cam2[2]) == 1 and abs(cam1[3] - cam2[3]) <= 1)
                or (abs(cam1[1] - cam2[1]) == 1 and abs(cam1[2] - cam2[2]) <= 1 and abs(cam1[3] - cam2[3]) <= 1)
            ):
                adj_matrix[i, j] = 1

# Print adjacency matrix

adj_matrix = np.array(adj_matrix)
with np.printoptions(threshold=np.inf):

    print(adj_matrix)
