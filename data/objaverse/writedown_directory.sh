#!/bin/bash

# 设置源目录和输出文件路径
SOURCE_DIR="/mnt/test/data/objaverse/rendered_2/"   # 替换为实际源目录路径
OUTPUT_FILE="/home/fudan248/zhangjinyu/code_repo/PoinTr/data/objaverse/v1.txt"  # 替换为实际输出文件路径

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "源目录 $SOURCE_DIR 不存在。"
    exit 1
fi

# 清空或创建输出文件
> "$OUTPUT_FILE"

# 遍历源目录下的每个子目录
for subdir in "$SOURCE_DIR"/*/; do
    # 获取子目录的名称
    subdir_name=$(basename "$subdir")
    
    # 写入子目录名称
    echo "${subdir_name}/" >> "$OUTPUT_FILE"
    
    # 遍历子目录下的每个孙目录
    for granddir in "$subdir"*/; do
        granddir_name=$(basename "$granddir")
        # 写入孙目录名称，使用缩进表示层级
        echo "    ${granddir_name}" >> "$OUTPUT_FILE"
    done
done

echo "目录结构已记录到 $OUTPUT_FILE"
