import numpy as np
import igs2gs.igs2gs_metrics.data_util as du
from pathlib import Path

camera_json = Path(
    "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/10-16-18-18_Dora_as-if-it-were-by-modigliani_42_5.0_0.5_2.0_0.2/30000/camera_8.json"
)

camera_data = du.load_camera_data(camera_json)
intrinsics = du.parse_intrinsic(camera_data)
extrinsics = du.parse_transform(camera_data["camera_to_world"])
# extrinsics = extrinsics[:3, :]


nose_3d = np.array([1.48455, -0.00642, -0.0161])
# nose_3d = np.array([1.372201, -0.001169, -0.099190])
nose_3d_homogeneous = np.append(nose_3d, 1)

# Apply the extrinsic matrix to transform the point to camera coordinates
# camera_coords = extrinsics @ nose_3d_homogeneous
camera_coords = np.linalg.inv(extrinsics) @ nose_3d_homogeneous

# Apply the intrinsic matrix to project the point to image coordinates
image_coords_homogeneous = intrinsics @ camera_coords[:3]

# Normalize the coordinates to get the 2D image coordinates
uv_coord = image_coords_homogeneous[:2] / image_coords_homogeneous[2]

print(uv_coord)
