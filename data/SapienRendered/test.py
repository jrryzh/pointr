import os
import random

sapien_dir = "/home/fudan248/zhangjinyu/code_repo/PoinTr/data/SapienRendered/sapien_output"

category_list = [d for d in os.listdir(sapien_dir) if os.path.isdir(os.path.join(sapien_dir, d))]

result_list = []
fail_list = []
for category in category_list:
    instance_list = os.listdir(os.path.join(sapien_dir, category))
    all, success, fail = 0, 0, 0
    for instance in instance_list:
        all += 1
        if os.path.exists(os.path.join(sapien_dir, category, instance, '0499_pcd.obj')) and os.path.exists(os.path.join(sapien_dir, category, instance, '0299_pose.txt')):
            result_list.append(os.path.join(sapien_dir, category, instance))
        else:
            fail_list.append(os.path.join(sapien_dir, category, instance))
            fail += 1
    print(f"{category} fail: {fail}/{all}")

# 从train 按4:1 取obj
train_list = []
test_list = []
for idx, path in enumerate(result_list):
    if idx % 10 ==0:
        test_list.append(path)
    else:
        train_list.append(path)
    
# 保存result_list到文件
train_list_path = os.path.join("/home/fudan248/zhangjinyu/code_repo/PoinTr/data/SapienRendered", "500view_nocs_train_list.txt")
with open(train_list_path, 'w') as f:
    for item in train_list:
        f.write(f"{item}\n")

test_list_path = os.path.join("/home/fudan248/zhangjinyu/code_repo/PoinTr/data/SapienRendered", "500view_nocs_test_list.txt")
with open(test_list_path, 'w') as f:
    for item in test_list:
        f.write(f"{item}\n")

# 保存fail_list到文件
fail_list_path = os.path.join("/home/fudan248/zhangjinyu/code_repo/PoinTr/data/SapienRendered", "500view_nocs_fail_list.txt")
with open(fail_list_path, 'w') as f:
    for item in fail_list:
        f.write(f"{item}\n")

print("Results saved to result_list.txt and fail_list.txt")
    

    