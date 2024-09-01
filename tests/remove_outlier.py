import open3d as o3d
import numpy as np

def read_obj_as_numpy(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

# 文件路径
file_path = "/data/nas/zjy/code_repo/pointr/tests/input/obj_output/f9ecc6749a251c0249852b2ef384d236_1199_output.obj"

# 读取OBJ文件顶点信息到NumPy数组
obsv_pcd = read_obj_as_numpy(file_path)

# 使用NumPy数组创建Open3D点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(obsv_pcd)

# 检查点云是否成功加载
if pcd.is_empty():
    raise ValueError("Failed to load point cloud from OBJ file.")

# 应用统计滤波以去除离群点
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=3.0)

# 选择并保留内点
inlier_cloud = pcd.select_by_index(ind)

# 将离群点选择出来
outlier_cloud = pcd.select_by_index(ind, invert=True)

# 将内点和离群点赋予不同的颜色
inlier_cloud.paint_uniform_color([0, 1, 0])  # 绿色
outlier_cloud.paint_uniform_color([1, 0, 0])  # 红色

# 将内点和离群点合并
combined_cloud = inlier_cloud + outlier_cloud

# 保存处理后的点云为新的OBJ文件
o3d.io.write_point_cloud("output/filtered_model.ply", combined_cloud)

