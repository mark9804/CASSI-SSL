# CASSI-SSL 训练脚本使用说明

Todo

- [ ] https://github.com/mengziyi64/TSA-Net/tree/master/TSA_Net_realdata/Data

## 概述

这个脚本包自动处理 CASSI-SSL 自监督高光谱图像重建的完整训练流程，包括数据预处理和模型训练。

## 文件说明

- `run_train.sh` - 主要的训练脚本
- `prepare_data.py` - 数据预处理脚本
- `download_real_masks.py` - 真实CASSI mask数据下载脚本
- `test_mask_data.py` - mask数据测试和可视化脚本
- `check_setup.py` - 环境检查脚本
- `README_TRAINING.md` - 本说明文档

## 环境配置

```bash
uv sync
```

## 数据集要求

请确保你有以下数据集：

### CAVE 数据集
```
datasets/CAVE/
├── balloons_ms/
├── beads_ms/
├── cd_ms/
└── ... (其他场景目录)
```

### KAIST 数据集  
```
datasets/KAIST/
├── set00/
│   ├── V000/
│   ├── V001/
│   └── ...
├── set01/
└── ... (其他set目录)
```

## 真实CASSI Mask数据

本脚本包现在支持使用真实的CASSI mask数据，基于TSA-Net项目：

### 自动下载真实mask数据
```bash
uv run download_real_masks.py
```

这将下载：
- 真实的硬件校准mask (256x256)
- 多种优化的mask模式
- 支持光谱色散的3D mask

### 可用的Mask类型
1. **TSA-Net真实mask** - 来自实际CASSI硬件
2. **优化Hadamard mask** - 基于Hadamard矩阵的优化模式  
3. **结构化随机mask** - 具有空间结构的随机模式
4. **硬件启发mask** - 模拟真实制造缺陷的模式

### 测试Mask数据质量
```bash
uv run test_mask_data.py
```

这将验证并可视化所有下载的mask：
- 分析开孔率、边缘密度等特性
- 生成mask模式的可视化图像
- 比较不同mask的性能特征

## 运行方法

### 1. 简单运行（推荐）
```bash
./run_train.sh
```

*注：脚本会自动检查并下载真实mask数据*

### 2. 分步运行

如果你想分步执行，可以：

```bash
# 步骤1: 数据预处理
uv run prepare_data.py

# 步骤2: 训练
cd train_code
uv run train.py --template gap_net --gpu_id "0" --data_root "../datasets/" --outf "../exp/gap_net/"
```

## 自定义参数

你可以修改 `run_train.sh` 中的训练参数：

- `--gpu_id "0"` - 使用的GPU编号
- `--batch_size 5` - 批大小（根据显存调整）
- `--max_epoch 400` - 最大训练轮数
- `--learning_rate 0.0004` - 学习率

## 输出文件

训练完成后，输出文件将保存在：

- `exp/gap_net/YYYY_MM_DD_HH_MM_SS/model/` - 模型检查点
- `exp/gap_net/YYYY_MM_DD_HH_MM_SS/result/` - 测试结果
- `exp/gap_net/YYYY_MM_DD_HH_MM_SS/model/log.txt` - 训练日志

## 可能的问题和解决方法

### 1. 内存不足
如果遇到GPU内存不足，可以：
- 减小 `--batch_size` 参数（从5改为2或3）
- 在 `run_train.sh` 中修改对应行

### 2. 数据格式问题
如果你的数据格式与期望不同：
- 检查 `prepare_data.py` 中的文件扩展名设置
- 修改 `process_cave_scene` 或 `process_kaist_set` 函数

### 3. 找不到图像文件
确保：
- CAVE数据集中每个场景目录包含图像文件
- KAIST数据集中每个set的V***目录包含图像文件
- 图像格式为 png/jpg/tif/tiff

### 4. 训练过程中断
训练脚本支持从检查点恢复：
- 使用 `--pretrained_model_path` 参数指向之前的模型文件
- 修改 `run_train.sh` 中的相应行

## 监控训练进度

训练过程中，你可以：

1. 查看实时日志：
```bash
tail -f exp/gap_net/*/model/log.txt
```

2. 监控GPU使用情况：
```bash
nvidia-smi -l 1
```

## 预期训练时间

根据硬件配置：
- 单个RTX 3090: 约12-24小时（400个epoch）
- 单个RTX 4090: 约8-16小时（400个epoch）

## 注意事项

1. 确保有足够的存储空间（至少50GB用于数据和模型）
2. 确保CUDA和PyTorch已正确安装
3. 第一次运行会进行数据预处理，需要额外时间
4. 训练前60个epoch使用基础损失函数，之后启用谱低秩损失

## 依赖要求

确保安装了以下Python包：
```bash
uv add torch torchvision scipy numpy pillow scikit-image gdown
```

*注：gdown包用于下载真实CASSI mask数据*

如有问题，请检查 `train_code/train.py` 中的具体实现。 