from pathlib import Path
from PIL import Image
import os
import math

num = 34

# Path to the root of the repository
root = Path("/home/lucky/Desktop/ig2g/2024-07-09_16-03-08/")
key = "render"

target = Path("/home/lucky/Desktop/" + key + "/elf")

subfolders = [f.path for f in os.scandir(root) if f.is_dir()]
subfolders.sort()


for i, path in enumerate(subfolders):
    epoch = math.floor(i / num)
    subfolder = Path(path)
    # render_path = Path(path).glob("_render.png")
    # print(render_path)

    for file in os.listdir(subfolder):

        if file.endswith("_" + key + ".png"):
            folder = target / f"epoch_{epoch}"
            folder.mkdir(parents=True, exist_ok=True)
            render = Image.open(subfolder / file)
            render.save(folder / file)

    # for r in render_path:
    #     print(r)
    # render = Image.open(render_path)
    # folder = target / f"epoch_{epoch}"
    # folder.mkdir(parents=True, exist_ok=True)
    # render.save(folder / render_path.name)
