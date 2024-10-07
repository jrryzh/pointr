import pandas as pd
import os
import shutil

# 定义参数
source_dir = '/home/add_disk_e/objaverse/hf-objaverse-v1/glbs/'  # 替换为您的源文件目录
target_dir = '/home/fudan248/zhangjinyu/code_repo/PoinTr/data/objaverse/raw_data2'  # 替换为您的目标文件目录

# selected_class_names = [
# 'apple',
# 'armor',
# 'army_tank',
# 'ax',
# 'banana',
# 'barrel',
# 'bookcase',
# 'crate',
# 'dog',
# 'doughnut',
# 'drum_(musical_instrument)',
# 'fighter_jet',
# 'fireplug',
# 'fish',
# 'globe',
# 'heart',
# 'jacket',
# 'jeep',
# 'keg',
# 'lamppost',
# 'mallet',
# 'penguin',
# 'pickup_truck',
# 'pizza',
# 'police_cruiser',
# 'pony',
# 'pop_(soda)',
# 'pumpkin',
# 'race_car',
# 'seashell',
# 'shield',
# 'spider',
# 'teddy_bear',
# 'turtle',
# 'wine_bucket',
# ]

shapenet_classes = [
    "Airplane", "Antenna", "Bicycle", "Birdhouse", "Bookshelf", "Bottle", "Bowl", "Bus", 
    "Cabinet", "Camera", "Car", "Chair", "Clock", "Computer", "Cup", "Flower Pot", "Guitar", 
    "Helmet", "Lamp", "Laptop", "Microphone", "Mug", "Pistol", "Rocket", "Skateboard", 
    "Sofa", "Table", "Telephone", "Vase", "Wine Bottle", "Motorcycle", "Boat", "Train", 
    "Truck", "Fire Hydrant", "Traffic Light", "Bench", "Dresser", "Toilet", "Bed", "Sink", 
    "Refrigerator", "Microwave", "Oven", "Dishwasher", "Toaster", "Washing Machine", 
    "Dryer", "Vacuum Cleaner", "Heater", "Fan", "Air Conditioner", "Radiator", "Fireplace", 
    "Desk"
]
shapenet_classes = [c.lower() for c in shapenet_classes]

selected_class_names = shapenet_classes + os.listdir('/home/fudan248/zhangjinyu/code_repo/PoinTr/data/objaverse/raw_data')

input_csv = 'kiui_morethan20_lvis.csv'          # 前面生成的 CSV 文件
output_csv = 'filtered_mapped_output2.csv'  # 您希望保存的新 CSV 文件

# 读取已生成的 CSV 文件
df = pd.read_csv(input_csv, header=None, names=['SubsetID', 'ClassName', 'UniqueID'])

# 筛选出 ClassName 在 selected_class_names 列表中的行
# df_filtered = df[df['ClassName'].isin(selected_class_names)]
df_filtered = df[~df['ClassName'].isin(selected_class_names)]

# 定义一个函数来组合文件路径
def create_file_paths(row):
    subset_id = row['SubsetID']
    class_name = row['ClassName']  # 获取类别名称
    unique_id = row['UniqueID']
    filename = f"{unique_id}.glb"
    
    # 源文件路径保持不变，基于 SubsetID
    source_path = os.path.join(source_dir, subset_id, filename)
    
    # 目标文件路径基于 ClassName，而不是 SubsetID
    target_path = os.path.join(target_dir, class_name, filename)
    
    return pd.Series([source_path, target_path])

# 应用函数生成新的路径列
df_filtered[['SourcePath', 'TargetPath']] = df_filtered.apply(create_file_paths, axis=1)

# 选择需要保存的列
new_df = df_filtered[['SubsetID', 'UniqueID', 'ClassName', 'SourcePath', 'TargetPath']]

# 保存为新的 CSV 文件
new_df.to_csv(output_csv, index=False, header=True)

print(f"筛选完成，结果已保存为 {output_csv}")

# 可选：创建符号链接或复制文件

# 选择一种操作方式：符号链接或复制文件
create_symlinks = True  # 设置为 False 则进行复制文件

def ensure_dir(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if create_symlinks:
    # 创建符号链接
    for index, row in new_df.iterrows():
        source = row['SourcePath']
        target = row['TargetPath']
        
        # 检查源文件是否存在
        if not os.path.exists(source):
            print(f"源文件不存在: {source}")
            continue
        
        # 确保目标目录存在
        ensure_dir(target)
        
        try:
            if not os.path.exists(target):
                os.symlink(source, target)
                print(f"创建符号链接: {target} -> {source}")
            else:
                print(f"目标文件已存在: {target}")
        except Exception as e:
            print(f"无法创建符号链接 {target} -> {source}: {e}")
else:
    # 复制文件
    for index, row in new_df.iterrows():
        source = row['SourcePath']
        target = row['TargetPath']
        
        # 检查源文件是否存在
        if not os.path.exists(source):
            print(f"源文件不存在: {source}")
            continue
        
        # 确保目标目录存在
        ensure_dir(target)
        
        try:
            shutil.copy2(source, target)
            print(f"复制文件: {source} -> {target}")
        except Exception as e:
            print(f"无法复制文件 {source} -> {target}: {e}")
