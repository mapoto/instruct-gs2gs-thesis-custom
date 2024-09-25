import numpy as np
import cv2


def gs_matching(image0, image1, viewmat0, viewmat1, depth0, depth1, fx, fy, cx, cy):

    height, width = ..., ...  # Image properties
    fx, fy, cx, cy = ..., ..., ..., ...  # Camera intrinsics

    depth0 = ...  # Depth image of view0, shape H,W
    depth1 = ...  # Depth image of view1, shape H,W (not used)

    viewmat0 = ...  # Viewmat of view0, shape 4,4
    viewmat1 = ...  # Viewmat of view1, shape 4,4

    # Transformation matrix from view1 to view0
    transform = viewmat0 @ np.linalg.inv(viewmat1)

    # Pixel vector fields s.t. u1[x,y] = x, v1[x,y] = y
    u1 = np.arange(width).astype(np.float32)
    v1 = np.arange(height).astype(np.float32)
    u1, v1 = np.meshgrid(u1, v1)

    # H,W image -> H*W list of coordinates
    z1 = depth1.flatten()
    u1 = u1.flatten()
    v1 = v1.flatten()

    # u,v pixel coordinates -> x,y,z view1 coordinate system
    x1 = (u1 - cx) * z1 / fx
    y1 = (v1 - cy) * z1 / fy

    # XYZ in homogeneous coordinates
    xyz1_hom = np.stack((x1, y1, z1, np.ones_like(z1)), axis=1)

    # Move to view0 coordinate system
    xyz0_hom = xyz1_hom @ transform.T

    # Normalize homogeneous coordinate (should not be necessary)
    xyz0 = xyz0_hom[:, :3] / xyz0_hom[:, 3:4]

    # Split into X,Y,Z and move to u,v pixel coordinates
    z0 = xyz0[:, 2]
    u0 = xyz0[:, 0] * fx / z0 + cx
    v0 = xyz0[:, 1] * fy / z0 + cy

    # Make them into images again
    z0 = z0.reshape(height, width).astype(np.float32)
    u0 = u0.reshape(height, width).astype(np.float32)
    v0 = v0.reshape(height, width).astype(np.float32)

    # Example of using optical flow to map an image in view0 to view1:

    # Map the depth of view0 to view1
    # As long as all pixels visible in view0 are also visible in view1, these should (roughly) match up
    depth0_mapped = cv2.remap(depth0, u0, v0, cv2.INTER_CLOSEST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Compute MAE on valid pixels
    valid = depth0_mapped != 0
    error = np.mean(np.abs(depth0_mapped[valid] - depth1[valid]))
    print("Error:", error)
