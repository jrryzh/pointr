import trimesh
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SOURCEPATH = '/home/fudan248/zhangjinyu/code_repo/PoinTr/data/objaverse/raw_data2'
DESTPATH = '/home/add_disk/zhangjinyu/cvpr/normalized_data2'
MAX_WORKERS = 64  # 根据您的系统资源调整线程数量

def process_instance(category, path):
    """
    处理单个 .glb 文件：加载、缩放、平移并导出。
    """
    try:
        out_dir = os.path.join(DESTPATH, category)
        os.makedirs(out_dir, exist_ok=True)

        model_id = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, model_id) + '.glb'

        if os.path.exists(out_path):
            print(f'{model_id} already exists, skip')
            return f'Skipped {model_id}'

        mesh = trimesh.load(path, force='mesh')

        # 计算包围盒和缩放因子
        bbox = mesh.bounds
        bbox_size = bbox[1] - bbox[0]
        max_dim = bbox_size.max()
        scale_factor = 1.0 / max_dim

        # 缩放网格
        mesh.apply_scale(scale_factor)

        # 将网格平移到原点
        center = mesh.centroid
        mesh.apply_translation(-center)

        # 创建场景并导出
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name=model_id)
        scene.export(out_path)

        print(f'{path} -> {out_path}')
        return f'Processed {model_id}'
    except Exception as e:
        print(f'Error processing {path}: {e}')
        return f'Error {model_id}'

def main():
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for category in os.listdir(SOURCEPATH):
            category_path = os.path.join(SOURCEPATH, category)
            if not os.path.isdir(category_path):
                continue

            instance_paths = [
                os.path.join(category_path, item)
                for item in os.listdir(category_path)
                if item.endswith('.glb')
            ]

            for path in instance_paths:
                futures.append(executor.submit(process_instance, category, path))

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()  # 可选：处理返回结果

if __name__ == '__main__':
    main()
