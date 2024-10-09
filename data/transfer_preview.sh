#!/bin/bash

# 配置参数
LOCAL_DIR="/home/add_disk/zhangjinyu/cvpr/shapenet_render_data/"
SERVER_USER="zhangjinyu"
SERVER_IP="10.176.56.103"
SERVER_PORT=1207
SERVER_DIR="/nvme/zhangjinyu/shapenet_render_data/"
PARALLEL_JOBS=4
PIGZ_THREADS=8

echo "=============================="
echo "预先脚本：传输 Objaverse Render 数据"
echo "=============================="

# 检查本地目录是否存在
if [ ! -d "$LOCAL_DIR" ]; then
    echo "错误：本地目录不存在: $LOCAL_DIR"
    exit 1
fi

echo "本地目录存在: $LOCAL_DIR"

# 导航到本地目录
echo "导航到本地目录: $LOCAL_DIR"
echo "cd \"$LOCAL_DIR\""

# 列出所有子目录（用于压缩）
echo "将要压缩的子目录列表："
find "$LOCAL_DIR" -mindepth 1 -maxdepth 1 -type d
echo

# 预览压缩命令
echo "将执行以下并行压缩命令："
find "$LOCAL_DIR" -mindepth 1 -maxdepth 1 -type d | parallel -j "$PARALLEL_JOBS" "echo tar -cf - {} | pigz -p $PIGZ_THREADS > {.}.tar.gz"
echo

# 预览创建服务器目标目录的命令
echo "将执行以下命令以确保服务器目标目录存在："
echo "ssh -p $SERVER_PORT $SERVER_USER@$SERVER_IP \"mkdir -p '$SERVER_DIR'\""
echo

# 列出将要传输的压缩包
echo "将要传输的压缩包列表："
ls "$LOCAL_DIR"*.tar.gz 2>/dev/null || echo "没有找到压缩包（预期）"
echo

# 预览传输命令
echo "将执行以下并行传输命令："
ls "$LOCAL_DIR"*.tar.gz 2>/dev/null | parallel -j "$PARALLEL_JOBS" "echo rsync -avz -e \"ssh -p $SERVER_PORT\" {} $SERVER_USER@$SERVER_IP:\"$SERVER_DIR\""
echo

echo "=============================="
echo "预先脚本执行完毕。"
echo "请检查上述命令是否正确，然后移除 'echo' 前缀以执行实际操作。"
echo "例如，将 'echo tar ...' 修改为 'tar ...'。"
echo "=============================="
