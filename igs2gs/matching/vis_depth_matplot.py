import numpy as np
import matplotlib.pyplot as plt
import torch

# Example depth map (for illustration purposes)
depth_map = np.random.rand(480, 640) * 10  # Random depth values between 0 and 10


depth_map = (
    torch.load(
        "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/igs2gs/10-16-18-18_Dora_as-if-it-were-by-modigliani_42_5.0_0.5_2.0_0.2/30000/8_depth.pt",
        weights_only=True,
    )
    .detach()
    .cpu()
)

maxima = torch.max(depth_map)
mask = depth_map == maxima
depth_map[mask] = 0
print(maxima)

# Plotting the depth map
plt.figure(figsize=(8, 6))
plt.imshow(depth_map, cmap="viridis")
plt.colorbar(label="Depth (units)")
plt.title("Depth Map Visualization")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.show()
