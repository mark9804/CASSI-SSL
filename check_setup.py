#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CASSI-SSL 环境检查脚本
验证训练环境是否就绪
"""

import os
import sys
import importlib


def check_python_packages():
    """检查必要的Python包"""
    required_packages = [
        "torch",
        "torchvision",
        "scipy",
        "numpy",
        "PIL",
        "skimage",
        "logging",
        "datetime",
        "gdown",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == "PIL":
                importlib.import_module("PIL")
            elif package == "skimage":
                importlib.import_module("skimage")
            else:
                importlib.import_module(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")

    return missing_packages


def check_cuda():
    """检查CUDA可用性"""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ CUDA 可用，检测到 {gpu_count} 个GPU")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
            return True
        else:
            print("✗ CUDA 不可用")
            return False
    except ImportError:
        print("✗ PyTorch 未安装，无法检查CUDA")
        return False


def check_datasets():
    """检查数据集目录"""
    datasets_ok = True

    # 检查CAVE数据集
    if os.path.exists("datasets/CAVE"):
        cave_scenes = [
            d
            for d in os.listdir("datasets/CAVE")
            if os.path.isdir(os.path.join("datasets/CAVE", d))
        ]
        print(f"✓ CAVE数据集目录存在，包含 {len(cave_scenes)} 个场景")
    else:
        print("✗ CAVE数据集目录不存在 (datasets/CAVE)")
        datasets_ok = False

    # 检查KAIST数据集
    if os.path.exists("datasets/KAIST"):
        kaist_sets = [
            d
            for d in os.listdir("datasets/KAIST")
            if d.startswith("set") and os.path.isdir(os.path.join("datasets/KAIST", d))
        ]
        print(f"✓ KAIST数据集目录存在，包含 {len(kaist_sets)} 个set")
    else:
        print("✗ KAIST数据集目录不存在 (datasets/KAIST)")
        datasets_ok = False

    return datasets_ok


def check_files():
    """检查必要的文件"""
    required_files = [
        "train_code/train.py",
        "train_code/option.py",
        "train_code/utils.py",
        "train_code/architecture.py",
        "train_code/loss.py",
        "run_train.sh",
        "prepare_data.py",
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} 存在")
        else:
            missing_files.append(file_path)
            print(f"✗ {file_path} 不存在")

    return missing_files


def check_directories():
    """检查和创建必要的目录"""
    required_dirs = ["exp", "exp/gap_net"]

    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ 目录 {dir_path} 存在")
        else:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ 已创建目录 {dir_path}")


def main():
    """主检查函数"""
    print("=" * 50)
    print("CASSI-SSL 环境检查")
    print("=" * 50)

    print("\n1. 检查Python包...")
    missing_packages = check_python_packages()

    print("\n2. 检查CUDA...")
    cuda_ok = check_cuda()

    print("\n3. 检查数据集...")
    datasets_ok = check_datasets()

    print("\n4. 检查必要文件...")
    missing_files = check_files()

    print("\n5. 检查目录...")
    check_directories()

    print("\n" + "=" * 50)
    print("检查结果汇总")
    print("=" * 50)

    all_ok = True

    if missing_packages:
        print(f"✗ 缺少的Python包: {', '.join(missing_packages)}")
        print("  安装命令: pip install " + " ".join(missing_packages))
        all_ok = False
    else:
        print("✓ 所有必要的Python包已安装")

    if cuda_ok:
        print("✓ CUDA环境正常")
    else:
        print("✗ CUDA环境有问题")
        all_ok = False

    if datasets_ok:
        print("✓ 数据集目录正常")
    else:
        print("✗ 数据集目录有问题")
        all_ok = False

    if missing_files:
        print(f"✗ 缺少的文件: {', '.join(missing_files)}")
        all_ok = False
    else:
        print("✓ 所有必要文件存在")

    print("\n" + "=" * 50)
    if all_ok:
        print("🎉 环境检查通过！可以开始训练了。")
        print("\n运行命令: ./run_train.sh")
    else:
        print("❌ 环境检查失败，请修复上述问题后重试。")
    print("=" * 50)


if __name__ == "__main__":
    main()
