import open3d as o3d


def load_and_visualize_ply(file_path):
    # Load the PLY file
    pcd = o3d.io.read_point_cloud(file_path)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    file_path = "point_cloud8_color.ply"  # Replace with your PLY file path
    load_and_visualize_ply(file_path)
