#!/bin/bash

# CASSI-SSL 训练脚本
# 用法: ./run_train.sh

echo "开始 CASSI-SSL 自监督训练..."

# 检查必要的目录是否存在
if [ ! -d "datasets/CAVE" ]; then
    echo "错误: 找不到 datasets/CAVE 目录"
    exit 1
fi

if [ ! -d "datasets/KAIST" ]; then
    echo "错误: 找不到 datasets/KAIST 目录"
    exit 1
fi

# 创建必要的目录
mkdir -p exp/gap_net/
mkdir -p datasets/masks/simulation/

# 数据预处理
echo "开始数据预处理..."
if [ ! -d "datasets/CAVE_processed" ] || [ ! -d "datasets/KAIST_processed" ]; then
    echo "运行数据预处理脚本..."
    uv run prepare_data.py
    if [ $? -ne 0 ]; then
        echo "错误: 数据预处理失败"
        exit 1
    fi
else
    echo "检测到已预处理的数据，跳过预处理步骤"
fi

# 下载并设置真实的CASSI mask数据
if [ ! -f "datasets/masks/simulation/mask.mat" ]; then
    echo "正在下载真实的CASSI mask数据..."
    uv run download_real_masks.py
    if [ $? -ne 0 ]; then
        echo "错误: 真实mask数据下载失败"
        exit 1
    fi
else
    echo "检测到已存在的mask文件，跳过下载"
fi

# 设置环境变量
export PYTHONPATH="${PWD}:${PWD}/train_code:${PYTHONPATH}"

# 进入训练代码目录
cd train_code

# 运行训练
uv run train.py \
    --template gap_net \
    --gpu_id "0" \
    --data_root "../datasets/" \
    --outf "../exp/gap_net/" \
    --method gap_net \
    --input_setting Y \
    --input_mask Phi_PhiPhiT \
    --batch_size 5 \
    --max_epoch 400 \
    --scheduler CosineAnnealingLR \
    --epoch_sam_num 1000 \
    --learning_rate 0.0004

echo "训练完成！" 