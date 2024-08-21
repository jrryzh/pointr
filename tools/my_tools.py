# import numpy as np

# def npy_to_obj(npy_file, obj_file):
#     # 加载 .npy 文件
#     data = np.load(npy_file)

#     # 确保数据的形状正确
#     if data.ndim != 2 or data.shape[1] != 3:
#         raise ValueError("The .npy file should contain nx3 array of points")

#     with open(obj_file, 'w') as f:
#         # 写入 .obj 文件的头部信息
#         f.write("# Created by npy_to_obj script\n")

#         # 将点云数据写入 .obj 文件
#         for point in data:
#             f.write("v {:.4f} {:.4f} {:.4f}\n".format(point[0], point[1], point[2]))

#     print(f"Successfully converted {npy_file} to {obj_file}")

# if __name__ == "__main__":
#     # 输入 .npy 文件路径
#     npy_file = '/home/fudan248/zhangjinyu/code_repo/PoinTr/data/ShapeNet55-34/shapenet_pc/03001627-c12da8acb2c7973597e755dddca14449.npy'
#     # 输出 .obj 文件路径 
#     obj_file = '/home/fudan248/zhangjinyu/code_repo/PoinTr/tmp/file.obj'
    
#     npy_to_obj(npy_file, obj_file)
import numpy as np
import open3d
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd', '.ply']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.obj']:
            return cls._read_obj(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    
    @classmethod
    def _read_obj(cls, file_path):
        vertices = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.strip().split()[1:4])))
        return np.array(vertices)
    
categories = {
    '02691156': 'airplane', 
    '02747177': 'ashcan',
    '02773838': 'bag',
    '02801938': 'basket',
    '02808440': 'bathtub',
    '02818832': 'bed',
    '02828884': 'bench', 
    '02843684': 'birdhouse', 
    '02871439': 'bookshelf', 
    '02876657': 'bottle', 
    '02880940': 'bowl',
    '02924116': 'bus', 
    '02933112': 'cabinet',
    '02942699': 'camera',
    '02946921': 'can', 
    '02954340': 'cap', 
    '02958343': 'car',
    '03001627': 'chair', 
    '03046257': 'clock', 
    '03085013': 'keypad', 
    '03207941': 'dishwasher', 
    '03211117': 'display', 
    '03261776': 'earphone', 
    '03325088': 'faucet', 
    '03337140': 'file',
    '03467517': 'guitar', 
    '03513137': 'helmet',
    '03593526': 'jar', 
    '03624134': 'knife',
    '03636649': 'lamp', 
    '03642806': 'laptop', 
    '03691459': 'loudspeaker', 
    '03710193': 'mailbox', 
    '03759954': 'microphone', 
    '03761084': 'microwave', 
    '03790512': 'motorcycle', 
    '03797390': 'mug', 
    '03928116': 'piano', 
    '03938244': 'pillow', 
    '03948459': 'pistol', 
    '03991062': 'pot', 
    '04004475': 'printer', 
    '04074963': 'remote', 
    '04090263': 'rifle', 
    '04099429': 'rocket', 
    '04225987': 'skateboard', 
    '04256520': 'sofa', 
    '04330267': 'stove', 
    '04379243': 'table', 
    '04401088': 'telephone', 
    '04460130': 'tower', 
    '04468005': 'train', 
    '04530566': 'vessel', 
    # '02834778': 'bike'
    '04554684': 'washer'
}

def load_pose(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            matrix.append([float(x) for x in line.strip().split()])
    return np.array(matrix).astype(np.float32)

def apply_transformation(vertices, transformation_matrix):
    num_vertices = vertices.shape[0]
    homogenous_vertices = np.hstack([vertices, np.ones((num_vertices, 1))])
    transformed_vertices = homogenous_vertices.dot(transformation_matrix.T)
    return transformed_vertices[:, :3]

def process_file(index, sample):
    try:
        data = {}
        data['partial'] = IO.get(sample['partial_pc_path']).astype(np.float32)
        pose = load_pose(sample['pose_path'])
        data['gt'] = apply_transformation(IO.get(sample['instance_path']).astype(np.float32), pose)
        
        # assert data['gt'].shape[0] > 8192
        # assert data['partial'].shape[0] > 2048
        print("gt shape:", data['gt'].shape)
        print("partial shape:", data['partial'].shape)
        
        return (index, data, None)  # Return index and data if succeeded
    except Exception as e:
        print(Exception(f"Exception occurred for index {index}: {e}"))
        return (index, None, sample)  # Return index and sample if failed

npoints = 8192
paritial_points_path = "/mnt/test/data/shapenet/shapenetcorev2_render_output2/"
instance_path = "/mnt/test/data/shapenet/flipped/"
file_list = []
for key, value in categories.items():
    id_list = os.listdir(os.path.join(paritial_points_path, value))
    for id in id_list:
        rendering_path = os.path.join(paritial_points_path, value, id)
        for i in [1, 299]:
            file_list.append({
                'taxonomy_id': key,
                'model_id': id,
                'partial_pc_path': os.path.join(rendering_path, f'{i:04}_pcd.obj'),
                'pose_path': os.path.join(rendering_path, f'{i:04}_pose.txt'),
                'instance_path': os.path.join(instance_path, key, id, 'models/model_normalized.obj')
            })    

print(len(file_list))
failed_lst = []

# for idx in range(len(file_list)):
#     if idx%100 == 0:
#         print(f"Processing {idx}/{len(file_list)}")
#     try:
#         sample = file_list[idx]
#         data = {}

#         data['partial'] = IO.get(sample['partial_pc_path']).astype(np.float32)
#         pose = load_pose(sample['pose_path'])
#         data['gt'] = apply_transformation(IO.get(sample['instance_path']).astype(np.float32), pose)
#     except:
#         failed_lst.append(sample)
#         continue
# Number of workers can be adjusted based on the system's capabilities
max_workers = 8

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_file, idx, sample): idx for idx, sample in enumerate(file_list)}
    for future in as_completed(futures):
        idx = futures[future]
        try:
            index, data, failed_sample = future.result()
            if index % 100 == 0:
                print(f"Processing {index}/{len(file_list)}")
            if failed_sample:
                failed_lst.append(failed_sample)
        except Exception as e:
            print(f"Exception occurred for index {idx}: {e}")

# save
# failed_lst.sort(key=lambda x: x['taxonomy_id'])
# with open('failed_lst.txt', 'w') as f:
#     for sample in failed_lst:
#         f.write(f"{sample['taxonomy_id']}/{sample['model_id']}\n")

    # assert data['gt'].shape[0] == npoints
    # if data['gt'].shape[0] != npoints:
    #     print(f"Skipping {sample['taxonomy_id']}/{sample['model_id']} due to incomplete point cloud, {data['gt'].shape[0]} instead of {npoints}")
    #     continue

failed_lst = """
02691156/e7e73007e0373933c4c280b3db0d6264
02691156/e7e73007e0373933c4c280b3db0d6264
02691156/e7e73007e0373933c4c280b3db0d6264
02691156/e7e73007e0373933c4c280b3db0d6264
02691156/e7e73007e0373933c4c280b3db0d6264
02691156/e7e73007e0373933c4c280b3db0d6264
02691156/e7e73007e0373933c4c280b3db0d6264
02691156/e7e73007e0373933c4c280b3db0d6264
02691156/e7e73007e0373933c4c280b3db0d6264
02691156/e7e73007e0373933c4c280b3db0d6264
02747177/cf158e768a6c9c8a17cab8b41d766398
02747177/cf158e768a6c9c8a17cab8b41d766398
02747177/cf158e768a6c9c8a17cab8b41d766398
02747177/cf158e768a6c9c8a17cab8b41d766398
02747177/cf158e768a6c9c8a17cab8b41d766398
02747177/cf158e768a6c9c8a17cab8b41d766398
02747177/cf158e768a6c9c8a17cab8b41d766398
02747177/cf158e768a6c9c8a17cab8b41d766398
02747177/cf158e768a6c9c8a17cab8b41d766398
02747177/cf158e768a6c9c8a17cab8b41d766398
02773838/f5108ede5ca11f041f6736765dee4fa9
02773838/f5108ede5ca11f041f6736765dee4fa9
02773838/f5108ede5ca11f041f6736765dee4fa9
02773838/f5108ede5ca11f041f6736765dee4fa9
02773838/f5108ede5ca11f041f6736765dee4fa9
02773838/f5108ede5ca11f041f6736765dee4fa9
02773838/f5108ede5ca11f041f6736765dee4fa9
02773838/f5108ede5ca11f041f6736765dee4fa9
02773838/f5108ede5ca11f041f6736765dee4fa9
02773838/f5108ede5ca11f041f6736765dee4fa9
02801938/b02c92fb423f3251a6a37f69f7f8f4c7
02801938/b02c92fb423f3251a6a37f69f7f8f4c7
02801938/b02c92fb423f3251a6a37f69f7f8f4c7
02801938/b02c92fb423f3251a6a37f69f7f8f4c7
02801938/b02c92fb423f3251a6a37f69f7f8f4c7
02801938/b02c92fb423f3251a6a37f69f7f8f4c7
02801938/b02c92fb423f3251a6a37f69f7f8f4c7
02801938/b02c92fb423f3251a6a37f69f7f8f4c7
02808440/7e3f69072a9c288354d7082b34825ef0
02808440/7e3f69072a9c288354d7082b34825ef0
02808440/7e3f69072a9c288354d7082b34825ef0
02808440/7e3f69072a9c288354d7082b34825ef0
02808440/7e3f69072a9c288354d7082b34825ef0
02808440/7e3f69072a9c288354d7082b34825ef0
02808440/7e3f69072a9c288354d7082b34825ef0
02808440/7e3f69072a9c288354d7082b34825ef0
02808440/7e3f69072a9c288354d7082b34825ef0
02808440/7e3f69072a9c288354d7082b34825ef0
02818832/5f9dd5306ad6b3539867b7eda2e4d345
02818832/5f9dd5306ad6b3539867b7eda2e4d345
02818832/5f9dd5306ad6b3539867b7eda2e4d345
02818832/5f9dd5306ad6b3539867b7eda2e4d345
02818832/5f9dd5306ad6b3539867b7eda2e4d345
02818832/5f9dd5306ad6b3539867b7eda2e4d345
02818832/5f9dd5306ad6b3539867b7eda2e4d345
02818832/5f9dd5306ad6b3539867b7eda2e4d345
02818832/5f9dd5306ad6b3539867b7eda2e4d345
02818832/5f9dd5306ad6b3539867b7eda2e4d345
02828884/86ab9c42f10767d8eddca7e2450ee088
02828884/86ab9c42f10767d8eddca7e2450ee088
02828884/86ab9c42f10767d8eddca7e2450ee088
02828884/86ab9c42f10767d8eddca7e2450ee088
02828884/86ab9c42f10767d8eddca7e2450ee088
02828884/86ab9c42f10767d8eddca7e2450ee088
02828884/86ab9c42f10767d8eddca7e2450ee088
02828884/86ab9c42f10767d8eddca7e2450ee088
02828884/86ab9c42f10767d8eddca7e2450ee088
02828884/86ab9c42f10767d8eddca7e2450ee088
02843684/e2ae1407d8f26bba7a1a3731b05e0891
02843684/e2ae1407d8f26bba7a1a3731b05e0891
02843684/e2ae1407d8f26bba7a1a3731b05e0891
02843684/e2ae1407d8f26bba7a1a3731b05e0891
02843684/e2ae1407d8f26bba7a1a3731b05e0891
02843684/e2ae1407d8f26bba7a1a3731b05e0891
02843684/e2ae1407d8f26bba7a1a3731b05e0891
02843684/e2ae1407d8f26bba7a1a3731b05e0891
02843684/e2ae1407d8f26bba7a1a3731b05e0891
02843684/e2ae1407d8f26bba7a1a3731b05e0891
02871439/82b88ee820dfb00762ad803a716d1873
02871439/82b88ee820dfb00762ad803a716d1873
02871439/82b88ee820dfb00762ad803a716d1873
02871439/82b88ee820dfb00762ad803a716d1873
02871439/82b88ee820dfb00762ad803a716d1873
02871439/82b88ee820dfb00762ad803a716d1873
02871439/82b88ee820dfb00762ad803a716d1873
02871439/82b88ee820dfb00762ad803a716d1873
02871439/82b88ee820dfb00762ad803a716d1873
02871439/82b88ee820dfb00762ad803a716d1873
02876657/b7ffc4d34ffbd449940806ade53ef2f
02876657/b7ffc4d34ffbd449940806ade53ef2f
02876657/b7ffc4d34ffbd449940806ade53ef2f
02876657/b7ffc4d34ffbd449940806ade53ef2f
02876657/b7ffc4d34ffbd449940806ade53ef2f
02876657/b7ffc4d34ffbd449940806ade53ef2f
02876657/b7ffc4d34ffbd449940806ade53ef2f
02876657/b7ffc4d34ffbd449940806ade53ef2f
02876657/b7ffc4d34ffbd449940806ade53ef2f
02876657/b7ffc4d34ffbd449940806ade53ef2f
"""

failed_set = set(failed_lst.split())

print(len(failed_set))
print(failed_set)