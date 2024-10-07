#!/bin/bash

# 设置源目录和目标目录
SOURCE_DIR="/mnt/test/data/objaverse/rendered_2/"   # 替换为实际源目录路径
TARGET_DIR="/home/fudan248/zhangjinyu/code_repo/PoinTr/data/objaverse/000rgb/"   # 替换为实际目标目录路径

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "源目录 $SOURCE_DIR 不存在。"
    exit 1
fi

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 遍历源目录下的每个子目录
for subdir in "$SOURCE_DIR"/*/; do
    # 获取子目录的名称
    subdir_name=$(basename "$subdir")
    
    # 遍历子目录下的每个孙目录
    for granddir in "$subdir"*/; do
        # 获取孙目录的名称
        granddir_name=$(basename "$granddir")
        
        # 定义源文件路径
        src_file="${granddir}0000_rgb.png"
        
        # 检查源文件是否存在
        if [ -f "$src_file" ]; then
            # 创建目标子目录（如果不存在）
            mkdir -p "$TARGET_DIR/$subdir_name"
            
            # 定义目标文件名
            dest_file="$TARGET_DIR/$subdir_name/${granddir_name}.png"
            
            # 复制并重命名文件
            cp "$src_file" "$dest_file"
            
            echo "已复制 $src_file 到 $dest_file"
        else
            echo "文件 $src_file 不存在，跳过。"
        fi
    done
done
