import pandas as pd
import json
import objaverse

# 读取原始CSV文件
input_csv = 'kiuisobj_v1_merged_80K.csv'  # 替换为您的原始CSV文件路径
df = pd.read_csv(input_csv, header=None, names=['SubsetID', 'UniqueID'])

# # 读入mapping文件
# with open('/home/fudan248/zhangjinyu/code_repo/PoinTr/data/objaverse/lvis_annotations.json', 'r') as f:
#     _dict = json.load(f)

# # 构建UniqueID到类名称的映射字典
# new_dict = {}
# for k, v in _dict.items():
#     for _v in v:
#         new_dict[_v] = k

# # 映射子集ID到类名称
# def map_subset_id(unique_id):
#     return new_dict.get(unique_id, "Unknown")

def map_subset_id(unique_id):
    return objaverse.load_annotations(unique_id)[unique_id]['categories']

df['ClassName'] = df['UniqueID'].apply(map_subset_id)

# 检查是否有未映射的UniqueID
unmapped = df[df['ClassName'] == "Unknown"]['UniqueID'].unique()
if len(unmapped) > 0:
    print("以下UniqueID未找到对应的类名称，请更新映射字典：")
    for uid in unmapped:
        print(uid)

# 过滤掉ClassName为"Unknown"的行
df_filtered = df[df['ClassName'] != "Unknown"]

# 按ClassName排序
df_sorted = df_filtered.sort_values(by='ClassName')

# 创建新的DataFrame，包含SubsetID、ClassName和UniqueID
new_df = df_sorted[['SubsetID', 'ClassName', 'UniqueID']]

import ipdb; ipdb.set_trace()
# 保存为新的CSV文件
output_csv = 'mapped_output.csv'  # 您希望保存的新CSV文件路径
new_df.to_csv(output_csv, index=False, header=False)

print(f"转换完成，新的CSV文件已保存为 {output_csv}")
