from collections import defaultdict

# 读取文件
file_path = '/home/zhangjinyu/code_repo/pointr/data/SapienRendered/500view_shapenet_train_list.txt'
with open(file_path, 'r') as f:
    lines = f.readlines()

# 创建一个字典来存储每个类别的记录
category_dict = defaultdict(list)

# 遍历每一行，并按类别进行分组
for line in lines:
    # 假设类别路径是行的前几部分，例如 '/data/SapienRendered/sapien_output/can/' 表示类别 'can'
    category = line.split('/')[-2]  # 提取类别
    category_dict[category].append(line)

# 保留每个类别的前50条记录
filtered_lines = []
for category, records in category_dict.items():
    filtered_lines.extend(records[:50])

# 将结果写入新文件
output_path = '/home/zhangjinyu/code_repo/pointr/data/SapienRendered/500view_shapenet_first50_train_list.txt'
with open(output_path, 'w') as f:
    f.writelines(filtered_lines)

print(f"Filtered data saved to {output_path}")
