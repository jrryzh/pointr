# 原始类别字典，包含英文名称和中文注释
_categories = {
    '02691156': 'airplane',      # 飞机
    '02747177': 'ashcan',        # 垃圾桶
    '02773838': 'bag',            # 包
    '02801938': 'basket',         # 篮子
    '02808440': 'bathtub',        # 浴缸
    '02818832': 'bed',            # 床
    '02828884': 'bench',          # 长凳
    '02843684': 'birdhouse',      # 鸟舍
    '02871439': 'bookshelf',      # 书架
    '02876657': 'bottle',         # 瓶子
    '02880940': 'bowl',           # 碗
    '02924116': 'bus',            # 公交车
    '02933112': 'cabinet',        # 橱柜
    '02942699': 'camera',         # 相机
    '02946921': 'can',            # 罐头
    '02954340': 'cap',            # 帽子
    '02958343': 'car',            # 汽车
    '03001627': 'chair',          # 椅子
    '03046257': 'clock',          # 时钟
    '03085013': 'keypad',         # 按键盘
    '03207941': 'dishwasher',     # 洗碗机
    '03211117': 'display',        # 显示器
    '03261776': 'earphone',       # 耳机
    '03325088': 'faucet',         # 水龙头
    '03337140': 'file',           # 文件
    '03467517': 'guitar',         # 吉他
    '03513137': 'helmet',         # 头盔
    '03593526': 'jar',            # 罐子
    '03624134': 'knife',          # 刀
    '03636649': 'lamp',           # 灯
    '03642806': 'laptop',         # 笔记本电脑
    '03691459': 'loudspeaker',    # 扩音器
    '03710193': 'mailbox',        # 邮箱
    '03759954': 'microphone',     # 麦克风
    '03761084': 'microwave',      # 微波炉
    '03790512': 'motorcycle',     # 摩托车
    '03797390': 'mug',            # 马克杯
    '03928116': 'piano',          # 钢琴
    '03938244': 'pillow',         # 枕头
    '03948459': 'pistol',         # 手枪
    '03991062': 'pot',            # 锅
    '04004475': 'printer',        # 打印机
    '04074963': 'remote',         # 遥控器
    '04090263': 'rifle',          # 步枪
    '04099429': 'rocket',         # 火箭
    '04225987': 'skateboard',     # 滑板
    '04256520': 'sofa',           # 沙发
    '04330267': 'stove',          # 炉子
    '04379243': 'table',          # 桌子
    '04401088': 'telephone',      # 电话
    '04460130': 'tower',          # 塔
    '04468005': 'train',          # 火车
    '04530566': 'vessel',         # 容器
    '02834778': 'bike',           # 自行车
    '04554684': 'washer',         # 洗衣机
}

# 1. 获取类别名称的顺序列表（按照字典中出现的顺序）
category_names = list(_categories.values()) # sorted(list(_categories.values()))

# 2. 创建类别到标签的映射（0, 1, 2, 3, ...）
category_to_label = {category: idx for idx, category in enumerate(category_names)}

# 3. 创建新的字典，将类别名称替换为对应的标签
categories_with_labels = {key: category_to_label[value] for key, value in _categories.items()}

# 打印结果
# print("类别到标签的映射（category_to_label）:")
# for category, label in category_to_label.items():
#     print(f"  '{category}': {label}")

# print("\n新的类别字典（categories_with_labels）:")
# for key, label in categories_with_labels.items():
#     print(f"  '{key}': {label}")

