import json
import os
import shutil

def copy_files(src_path, dest_path):

    # 如果是文件，则进行复制
    if os.path.isfile(src_path):
        shutil.copy2(src_path, dest_path)
    else:
        print(f"复制文件 {src_path} 到 {dest_path}失败")

with open('small_lvis_annotations.json', 'r') as f:
    small_dic = json.load(f)

with open('lvis_inverted_index.json', 'r') as f:
    invert_dic = json.load(f)

address = dict()

for key, values in small_dic.items():
    address[key] = dict()
    for value in values:
        dir = invert_dic[value]
        address[key][value] = os.path.join('/home/add_disk_e/objaverse/hf-objaverse-v1/glbs',dir, value+'.glb')


'''with open('address.json', 'w') as f:
    json.dump(address, f, indent=4)'''

# for every instance, copy the file in the coresponding directory to a specified directory

dir_path = '/home/fudan248/zhangjinyu/code_repo/objaverse/texts'

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for category, object_list in address.items():
    with open(os.path.join(dir_path, category+'.txt'), 'w') as f:
        for _, path in object_list.items():
            f.write(path)
            f.write('\n')
