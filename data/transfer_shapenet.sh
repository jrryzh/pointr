#!/bin/bash

# 配置参数
LOCAL_DIR="/home/add_disk/zhangjinyu/cvpr/shapenet_render_data/"
SERVER_USER="zhangjinyu"
SERVER_IP="10.176.56.103"
SERVER_PORT=1207
SERVER_DIR="/nvme/zhangjinyu/shapenet_render_data/"
PARALLEL_JOBS=32
PIGZ_THREADS=16

# 导航到本地目录
cd "$LOCAL_DIR" || { echo "目录不存在: $LOCAL_DIR"; exit 1; }

# 压缩所有子目录
echo "开始压缩子目录..."
find . -mindepth 1 -maxdepth 1 -type d | parallel -j "$PARALLEL_JOBS" "tar -cf - {} | pigz -p $PIGZ_THREADS > {.}.tar.gz"
echo "压缩完成。"

# 创建服务器目标目录
echo "确保服务器目标目录存在..."
ssh -p "$SERVER_PORT" "$SERVER_USER@$SERVER_IP" "mkdir -p '$SERVER_DIR'"

# 传输压缩包
echo "开始传输压缩包..."
ls *.tar.gz | parallel -j "$PARALLEL_JOBS" "rsync -avz -e 'ssh -p $SERVER_PORT' {} $SERVER_USER@$SERVER_IP:'$SERVER_DIR'"
echo "传输完成。"
