import pandas as pd

# 定义输入CSV文件的路径
input_csv = 'mapped_output.csv'  # 请将此路径替换为您的mapped_output.csv文件的实际路径

# 定义列名称
column_names = ['SubsetID', 'ClassName', 'UniqueID']

# 读取CSV文件，没有标题行，因此设置header=None，并指定列名
try:
    df = pd.read_csv(input_csv, header=None, names=column_names)
except FileNotFoundError:
    print(f"文件未找到: {input_csv}")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"文件为空: {input_csv}")
    exit(1)
except Exception as e:
    print(f"读取CSV文件时发生错误: {e}")
    exit(1)

# 检查必要的列是否存在
required_columns = ['SubsetID', 'ClassName', 'UniqueID']
for col in required_columns:
    if col not in df.columns:
        print(f"缺少必要的列: {col}")
        exit(1)

# 计算每个ClassName的实例数量
class_counts = df['ClassName'].value_counts()

# 筛选出实例数量大于等于30的ClassName
filtered_class_names = class_counts[class_counts >= 30].index.tolist()

# 提取筛选后的唯一ClassName列表
unique_class_names_list = filtered_class_names

# 输出筛选后的唯一ClassName列表
print("筛选后，实例数量大于等于30的唯一ClassName列表如下：")
for class_name in unique_class_names_list:
    print(class_name)

# 已有结果，通过ls获得
finished_lst = [
    'aerosol_can', 'candle_holder', 'easel', 'hamper', 'houseboat', 'kite', 'lightbulb',
    'alligator', 'cart', 'envelope', 'handbag', 'hummingbird', 'kitten', 'lion',
    'ambulance', 'carton', 'Ferris_wheel', 'hand_glass', 'icecream', 'kiwi_fruit', 'lip_balm',
    'armchair', 'cash_register', 'ferry', 'hardback_book', 'ice_skate', 'knife', 'lizard',
    'armoire', 'casserole', 'fire_extinguisher', 'harmonium', 'inhaler', 'knitting_needle', 'locker',
    'ashtray', 'cassette', 'flowerpot', 'hat', 'inkpad', 'koala', 'lollipop',
    'automatic_washer', 'Christmas_tree', 'forklift', 'helicopter'
]

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

# 创建一个已完成的类别集合
finished_set = set(finished_lst + shapenet_classes)

# 初始化列表以保存结果
appeared = []
not_appeared = []

# 遍历筛选后的唯一ClassName列表，并分类
for item in unique_class_names_list:
    if item in finished_set:
        appeared.append(item)
    else:
        not_appeared.append(item)

# 打印结果
# print("Items that appeared in the first list:")
# for item in appeared:
#     print(f"- {item}")

print("\n实例数量大于等于30且未出现在已完成列表中的ClassName如下：")
for item in sorted(not_appeared):
    print(f"- {item}")

# 可选：如果需要同时查看出现在已完成列表中的ClassName，可以取消下面的注释
# print("\n实例数量大于等于30且已出现在已完成列表中的ClassName如下：")
# for item in sorted(appeared):
#     print(f"- {item}")
