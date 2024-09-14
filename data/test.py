import os

sapien_dir = "/data/nas/zjy/code_repo/pointr/data/SapienRendered"

category_list = os.listdir(sapien_dir)

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
    
# 保存result_list到文件
result_list_path = os.path.join("/data/nas/zjy/code_repo/pointr/data/SapienRendered", "500view_nocs_train_list.txt")
with open(result_list_path, 'w') as f:
    for item in result_list:
        f.write(f"{item}\n")

# 保存fail_list到文件
fail_list_path = os.path.join("/data/nas/zjy/code_repo/pointr/data/SapienRendered", "500view_nocs_fail_list.txt")
with open(fail_list_path, 'w') as f:
    for item in fail_list:
        f.write(f"{item}\n")

print("Results saved to result_list.txt and fail_list.txt")
    

    