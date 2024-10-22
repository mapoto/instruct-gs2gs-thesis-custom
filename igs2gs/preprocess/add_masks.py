import sys
import json
from pathlib import Path
import cv2


if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} path_to/mask.png path_to_transforms.json")
    sys.exit(1)
with open(sys.argv[1]) as input_file:
    file_contents = input_file.read()
parsed_json = json.loads(file_contents)

mask_folder = Path(sys.argv[1])
output_folder = Path(sys.argv[2])
backup_json = parsed_json.copy()

for frame in parsed_json["frames"]:
    name = Path(frame["file_path"]).stem

    # Add mask path to the frame
    frame["mask_path"] = f"masks/{name}.png"

    # Get the camera label
    camera_label = frame["camera_label"]

    # Get masks per frame
    mask_path = mask_folder / f"{camera_label}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape[:2]

    # Copy and downscale the masks
    processed_data_dir = output_folder.parent
    downscale_factors = [2, 4, 8]

    mask_path_1 = processed_data_dir / "masks"
    mask_path_1.mkdir(exist_ok=True)
    mask_path_1 = mask_path_1 / f"{name}.png"
    cv2.imwrite(str(mask_path_1), mask)
    print(f"Wrote {mask_path_1}")

    for downscale in downscale_factors:
        mask_path_i = processed_data_dir / f"masks_{downscale}"
        mask_path_i.mkdir(exist_ok=True)
        mask_path_i = mask_path_i / f"{name}.png"
        mask_i = cv2.resize(mask, (width // downscale, height // downscale), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(mask_path_i), mask_i)
        print(f"Wrote {mask_path_i}")

with open(sys.argv[2], "w") as output_file1:
    json.dump(parsed_json, output_file1, indent=4)

backup_path = Path(sys.argv[2]).with_suffix(".backup.json")
with open(backup_path, "w") as output_file2:
    json.dump(backup_json, output_file2, indent=4)
