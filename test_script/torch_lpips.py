import torch
import numpy as np
import cv2
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pathlib import Path

lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

# LPIPS needs the images to be in the [-1, 1] range.

root = Path(
    "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/10-22-15-02_Ephra_turn-him-into-Tolkien-Elf_42_5.0_0.5_2.0_0.2/30001/"
)


img1 = []
img2 = []

names1 = []
names2 = []

for item in root.glob("*.[jpJP][npNP]*[gG$]"):

    if "edited" in item.name:
        names1.append(item)
    if "original" in item.name:
        names2.append(item)

names1.sort()
names2.sort()
print(names1)
print(names2)
for item in names1:
    m = np.array(cv2.imread(item))
    m = (m / 255) - 1
    img1.append(m)
    break

for item in names2:
    m = np.array(cv2.imread(item))
    m = (m / 255) - 1
    img2.append(m)
    break

print(len(img1))
print(len(img2))

img1 = torch.from_numpy(np.array(img1)).permute(0, 3, 1, 2).float()
img2 = torch.from_numpy(np.array(img2)).permute(0, 3, 1, 2).float()

val = lpips(img1, img2)

mask_path = Path(
    "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/10-22-15-02_Ephra_turn-him-into-Tolkien-Elf_42_5.0_0.5_2.0_0.2/30001/0/depth_intersection_1to0.png"
)
mask = cv2.imread(mask_path)
mask = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).float()


img1 = cv2.imread(
    Path(
        "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs_ori/Turn him into Tolkien elf/37500/0_render.png"
    )
)
img1 = (img1 / 255) - 1
img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()

img2 = cv2.imread(
    Path(
        "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs_ori/Turn him into Tolkien elf/32500/0_render.png"
    )
)
img2 = (img2 / 255) - 1
img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()

val = lpips(img1, img2)

pips = (val * mask).flatten(1).sum(-1)
pips = pips / mask.flatten(1).sum(-1)


print(val)
print(pips)
