#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试和可视化CASSI mask数据的脚本
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import glob


def load_mask_file(mask_path: str) -> np.ndarray:
    """
    加载mask文件

    Args:
        mask_path: mask文件路径

    Returns:
        np.ndarray: mask数据
    """
    try:
        data = sio.loadmat(mask_path)
        # 尝试不同的可能的key名称
        possible_keys = ["mask", "mask_3d_shift", "Phi", "phi"]

        for key in possible_keys:
            if key in data:
                return data[key].astype(np.float32)

        # 如果没有找到标准key，返回第一个非系统key
        keys = [k for k in data.keys() if not k.startswith("__")]
        if keys:
            return data[keys[0]].astype(np.float32)

        raise ValueError("无法找到mask数据")

    except Exception as e:
        print(f"加载mask文件失败: {e}")
        return None


def analyze_mask(mask: np.ndarray, name: str) -> Dict:
    """
    分析mask的统计特性

    Args:
        mask: mask数据
        name: mask名称

    Returns:
        Dict: 分析结果
    """
    if mask is None:
        return {}

    if mask.ndim == 3:
        # 3D mask，分析第一个通道
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    stats = {
        "name": name,
        "shape": mask.shape,
        "dtype": mask.dtype,
        "min_value": float(np.min(mask)),
        "max_value": float(np.max(mask)),
        "mean_value": float(np.mean(mask)),
        "aperture_ratio": float(np.mean(mask_2d > 0.5)),  # 开孔率
        "unique_values": len(np.unique(mask)),
        "is_binary": len(np.unique(mask)) == 2,
    }

    return stats


def visualize_mask(mask: np.ndarray, name: str, save_path: str = None):
    """
    可视化mask

    Args:
        mask: mask数据
        name: mask名称
        save_path: 保存路径
    """
    if mask is None:
        return

    if mask.ndim == 3:
        # 3D mask，显示前几个通道
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(min(6, mask.shape[2])):
            axes[i].imshow(mask[:, :, i], cmap="gray", vmin=0, vmax=1)
            axes[i].set_title(f"{name} - Channel {i}")
            axes[i].axis("off")

        # 隐藏多余的子图
        for j in range(min(6, mask.shape[2]), 6):
            axes[j].axis("off")

    else:
        # 2D mask
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 显示mask
        axes[0].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"{name} - Mask Pattern")
        axes[0].axis("off")

        # 显示统计直方图
        axes[1].hist(mask.flatten(), bins=50, alpha=0.7)
        axes[1].set_title(f"{name} - Value Distribution")
        axes[1].set_xlabel("Pixel Value")
        axes[1].set_ylabel("Frequency")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"可视化保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def test_mask_properties(mask: np.ndarray, name: str):
    """
    测试mask的物理特性

    Args:
        mask: mask数据
        name: mask名称
    """
    if mask is None:
        return

    if mask.ndim == 3:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    print(f"\n=== {name} 特性测试 ===")

    # 检查是否为二进制mask
    unique_vals = np.unique(mask_2d)
    is_binary = len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals
    print(f"二进制mask: {'是' if is_binary else '否'}")

    # 开孔率
    aperture_ratio = np.mean(mask_2d > 0.5)
    print(f"开孔率: {aperture_ratio:.3f}")

    # 空间频率分析（简单）
    fft_mask = np.fft.fft2(mask_2d)
    power_spectrum = np.abs(fft_mask) ** 2
    avg_power = np.mean(power_spectrum)
    print(f"平均频谱功率: {avg_power:.2e}")

    # 连通域分析
    if is_binary:
        from scipy import ndimage

        labeled, num_features = ndimage.label(mask_2d)
        print(f"连通域数量: {num_features}")

    # 边缘密度
    edge_count = np.sum(np.abs(np.diff(mask_2d, axis=0))) + np.sum(
        np.abs(np.diff(mask_2d, axis=1))
    )
    edge_density = edge_count / (mask_2d.shape[0] * mask_2d.shape[1])
    print(f"边缘密度: {edge_density:.3f}")


def compare_masks(mask_dir: str):
    """
    比较不同的mask

    Args:
        mask_dir: mask文件目录
    """
    mask_files = glob.glob(os.path.join(mask_dir, "*.mat"))

    if not mask_files:
        print(f"在 {mask_dir} 中未找到mask文件")
        return

    print(f"找到 {len(mask_files)} 个mask文件")

    all_stats = []

    for mask_file in mask_files:
        name = os.path.basename(mask_file).replace(".mat", "")
        mask = load_mask_file(mask_file)

        if mask is not None:
            stats = analyze_mask(mask, name)
            all_stats.append(stats)

            print(f"\n=== {name} ===")
            print(f"形状: {stats['shape']}")
            print(f"数据类型: {stats['dtype']}")
            print(f"值范围: [{stats['min_value']:.3f}, {stats['max_value']:.3f}]")
            print(f"平均值: {stats['mean_value']:.3f}")
            print(f"开孔率: {stats['aperture_ratio']:.3f}")
            print(f"唯一值数量: {stats['unique_values']}")
            print(f"是否二进制: {'是' if stats['is_binary'] else '否'}")

            # 测试物理特性
            test_mask_properties(mask, name)

            # 可视化
            vis_path = f"mask_visualization_{name}.png"
            visualize_mask(mask, name, vis_path)

    # 创建比较表格
    if len(all_stats) > 1:
        print("\n" + "=" * 80)
        print("MASK比较表")
        print("=" * 80)
        print(f"{'名称':<20} {'形状':<15} {'开孔率':<8} {'二进制':<8} {'唯一值':<8}")
        print("-" * 80)

        for stats in all_stats:
            print(
                f"{stats['name']:<20} {str(stats['shape']):<15} {stats['aperture_ratio']:<8.3f} "
                f"{'是' if stats['is_binary'] else '否':<8} {stats['unique_values']:<8}"
            )


def main():
    """主函数"""
    print("CASSI Mask数据测试工具")
    print("=" * 50)

    mask_dir = "datasets/masks/simulation"

    if not os.path.exists(mask_dir):
        print(f"错误: mask目录不存在 ({mask_dir})")
        print("请先运行 python3 download_real_masks.py")
        return

    # 测试和比较所有mask
    compare_masks(mask_dir)

    print(f"\n可视化图像已保存到当前目录")
    print(f"所有mask文件位于: {mask_dir}")


if __name__ == "__main__":
    main()
