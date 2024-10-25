import re
import numpy as np
import os
from pathlib import Path
import json

image_root_path = Path("/home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Ephra_grn/images/")
view_dict_path = Path("/home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Ephra_grn/view_dict.json")


# Load images and view dictionary
images = [image_path.name for image_path in image_root_path.glob("*.[jpJP][npNP]*[gG$]")]
view_dict = json.load(open(view_dict_path, "r"))


# Parse camera names
def parse_camera_name(name):
    match = re.match(r"(\d+)-([1-3BC])-([1-6])-([1-2])-\d+-\d", name)
    if match:
        session, row, bar, position = match.groups()
        row = {"3": 1, "2": 2, "1": 3, "C": 4, "B": 5}[row]
        return (int(session), row, int(bar), int(position))
    return None


# Create a list of parsed camera details
cameras = [parse_camera_name(name) for name in view_dict.keys()]

# Create adjacency matrix
num_cameras = len(cameras)
adj_matrix = np.zeros((num_cameras, num_cameras), dtype=int)

# Determine adjacency
for i, cam1 in enumerate(cameras):
    for j, cam2 in enumerate(cameras):
        if i != j:
            # Check if cameras are adjacent

            condition1 = cam1[1] == cam2[1] and cam1[2] == cam2[2] and abs(cam1[3] - cam2[3]) <= 1
            condition2 = cam1[1] == cam2[1] and abs(cam1[2] - cam2[2]) == 1 and abs(cam1[3] - cam2[3]) <= 1
            condition3 = abs(cam1[1] - cam2[1]) == 1 and abs(cam1[2] - cam2[2]) <= 1 and abs(cam1[3] - cam2[3]) <= 1

            if i == 2:
                if condition1:
                    print("condition1", cam1, cam2)
                if condition2:
                    print("condition2", cam1, cam2)
                if condition3:
                    print("condition3", cam1, cam2)

            if condition1 or condition2 or condition3:
                # Ensure rows "1" and "C" are not considered adjacent
                if not ((cam1[1] == 1 and cam2[1] == 5) or (cam1[1] == 5 and cam2[1] == 1)):
                    adj_matrix[i, j] = 1

# Print adjacency matrix
adj_matrix = np.array(adj_matrix)
# with np.printoptions(threshold=np.inf):
#     print(adj_matrix)

# Save adjacency matrix as a CSV file
np.savetxt("adj.csv", adj_matrix, delimiter=",", fmt="%d")
