"""
Removes greenscreen from an image.
Usage: python greenscreen_remove.py image.jpg
"""

from PIL import Image
import numpy as np
import colorsys
from pathlib import Path

# Define the range of green color in HSV
GREEN_RANGE_MIN_HSV = (58, 15, 40)
GREEN_RANGE_MAX_HSV = (155, 100, 100)


def rgb_to_hsv_vectorized(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    hsv = np.vectorize(colorsys.rgb_to_hsv)(r, g, b)
    return hsv[0] * 360, hsv[1] * 100, hsv[2] * 100


def main():
    image_root_path = Path("/home/lucky/dataset/metashape_aligned/Irene_grn/masked/")
    root_output_path = image_root_path.parent / "clean"

    print(f"Output path: {root_output_path.as_posix()}")
    root_output_path.mkdir(exist_ok=True)

    for image_path in image_root_path.glob("*.[jpJP][npNP]*[gG$]"):
        file_path = image_path.as_posix()
        name = image_path.stem
        print(f"Processing {name}")
        im = Image.open(file_path).convert("RGBA")
        pix = np.array(im)

        r, g, b, a = pix[:, :, 0], pix[:, :, 1], pix[:, :, 2], pix[:, :, 3]
        h, s, v = rgb_to_hsv_vectorized(r, g, b)

        min_h, min_s, min_v = GREEN_RANGE_MIN_HSV
        max_h, max_s, max_v = GREEN_RANGE_MAX_HSV

        mask = (min_h <= h) & (h <= max_h) & (min_s <= s) & (s <= max_s) & (min_v <= v) & (v <= max_v)
        pix[mask] = (0, 0, 0, 0)

        out_im = Image.fromarray(pix)
        outpath = root_output_path / (name + ".png")
        print(f"Saving as {outpath.as_posix()}")
        out_im.save(outpath)


if __name__ == "__main__":
    main()
