import json

with open('/mnt/test/data/shapenet/cluster_dict.json', 'r') as f:
    cluster_dict = json.load(f)
    
with open('./data/SapienRendered/500view_shapenet_frequent50_train_list.txt', 'w') as f:
    for key in cluster_dict:
        cate_dict = cluster_dict[key]
        sorted_keys = sorted(cate_dict, key=lambda k: len(cate_dict[k]), reverse=True)[:50]
    
        for k in sorted_keys:
            f.write(f'./data/SapienRendered/sapien_output/{key}/{cate_dict[k][0]}\n')
            # print(f'./data/SapienRendered/sapien_output/{k}/{cate_dict[k][0]}')
    
    
    

    

