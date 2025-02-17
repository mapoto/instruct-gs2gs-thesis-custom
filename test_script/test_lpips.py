import torch
import numpy as np
import cv2
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pathlib import Path

lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

# LPIPS needs the images to be in the [-1, 1] range.

root = Path(
    "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/10-22-15-02_Ephra_turn-him-into-Tolkien-Elf_42_5.0_0.5_2.0_0.2/30001/0/"
)


img1 = []
comp = []

names1 = []

for item in root.glob("*.[jpJP][npNP]*[gG$]"):

    if "masked" in item.name:
        names1.append(item)

names1.sort()
print(names1)
for item in names1:
    m = np.array(cv2.imread(item))
    m = (m / 255) - 1
    img1.append(m)

im = np.array(cv2.imread("/home/lucky/Downloads/fail1.png")) / 255 - 1
im = cv2.resize(im, (512, 765))
comp.append(im)

img1 = torch.from_numpy(np.array(img1)).permute(0, 3, 1, 2).float()
comp = torch.from_numpy((np.array(comp))).permute(0, 3, 1, 2).float()
val = lpips(img1, comp)

print(val)
