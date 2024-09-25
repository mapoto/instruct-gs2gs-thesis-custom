from torchmetrics.image.fid import FrechetInceptionDistance
import torch

fid = FrechetInceptionDistance(feature=64)
# generate two slightly overlapping image intensity distributions
imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)



fid.update(imgs_dist1, real=True)
fid.update(imgs_dist2, real=False)
score = fid.compute()

print(score)
