#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本
将CAVE和KAIST数据集转换为训练代码期望的格式
"""

import os
import numpy as np
import scipy.io as sio
from PIL import Image
import glob
from typing import List, Tuple
import argparse


def process_cave_scene(scene_path: str, output_dir: str, scene_idx: int) -> bool:
    """
    处理单个CAVE场景

    Args:
        scene_path: 场景目录路径
        output_dir: 输出目录
        scene_idx: 场景编号

    Returns:
        bool: 处理是否成功
    """
    try:
        # 查找场景目录中的图像文件
        image_files = []
        for ext in ["*.png", "*.jpg", "*.tif", "*.tiff"]:
            image_files.extend(glob.glob(os.path.join(scene_path, ext)))

        if not image_files:
            # 检查是否有子目录
            subdirs = [
                d
                for d in os.listdir(scene_path)
                if os.path.isdir(os.path.join(scene_path, d))
            ]
            if subdirs:
                subdir_path = os.path.join(scene_path, subdirs[0])
                for ext in ["*.png", "*.jpg", "*.tif", "*.tiff"]:
                    image_files.extend(glob.glob(os.path.join(subdir_path, ext)))

        if not image_files:
            print(f"警告: 在 {scene_path} 中未找到图像文件")
            return False

        # 按文件名排序
        image_files.sort()

        # 读取图像并组合成高光谱图像
        images = []
        for img_file in image_files[:28]:  # 只取前28个波段
            img = Image.open(img_file).convert("L")  # 转为灰度
            img_array = np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]
            images.append(img_array)

        if len(images) < 28:
            # 如果波段不足28个，复制最后一个波段
            while len(images) < 28:
                images.append(images[-1])

        # 组合成3D高光谱图像 [H, W, C]
        hsi = np.stack(images, axis=2)

        # 调整大小到256x256（如果需要）
        if hsi.shape[0] != 256 or hsi.shape[1] != 256:
            from scipy.ndimage import zoom

            zoom_factors = (256 / hsi.shape[0], 256 / hsi.shape[1], 1)
            hsi = zoom(hsi, zoom_factors, order=1)

        # 扩展为期望的尺寸（可能需要更大的尺寸用于裁剪）
        if hsi.shape[0] < 512 or hsi.shape[1] < 512:
            # 将图像扩展到512x512以便训练时裁剪
            expanded = np.zeros((512, 512, 28), dtype=np.float32)
            h, w = hsi.shape[:2]
            start_h = (512 - h) // 2
            start_w = (512 - w) // 2
            expanded[start_h : start_h + h, start_w : start_w + w, :] = hsi
            hsi = expanded

        # 保存为.mat文件
        output_file = os.path.join(output_dir, f"scene{scene_idx:02d}.mat")
        sio.savemat(output_file, {"img_expand": (hsi * 65536).astype(np.uint16)})

        print(
            f"已处理场景 {scene_idx}: {os.path.basename(scene_path)} -> {output_file}"
        )
        return True

    except Exception as e:
        print(f"处理场景 {scene_path} 时出错: {e}")
        return False


def process_kaist_set(set_path: str, output_dir: str, set_name: str) -> bool:
    """
    处理KAIST数据集的一个set

    Args:
        set_path: set目录路径
        output_dir: 输出目录
        set_name: set名称

    Returns:
        bool: 处理是否成功
    """
    try:
        # 查找set中的视频目录
        video_dirs = [
            d
            for d in os.listdir(set_path)
            if d.startswith("V") and os.path.isdir(os.path.join(set_path, d))
        ]
        video_dirs.sort()

        for i, video_dir in enumerate(video_dirs):
            video_path = os.path.join(set_path, video_dir)

            # 查找图像文件
            image_files = []
            for ext in ["*.png", "*.jpg", "*.tif", "*.tiff"]:
                image_files.extend(glob.glob(os.path.join(video_path, ext)))

            if not image_files:
                continue

            image_files.sort()

            # 读取图像并组合成高光谱图像
            images = []
            for img_file in image_files[:28]:  # 只取前28个波段
                img = Image.open(img_file).convert("L")  # 转为灰度
                img_array = np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]
                images.append(img_array)

            if len(images) < 28:
                # 如果波段不足28个，复制最后一个波段
                while len(images) < 28:
                    images.append(images[-1])

            # 组合成3D高光谱图像 [H, W, C]
            hsi = np.stack(images, axis=2)

            # 调整大小到256x256
            if hsi.shape[0] != 256 or hsi.shape[1] != 256:
                from scipy.ndimage import zoom

                zoom_factors = (256 / hsi.shape[0], 256 / hsi.shape[1], 1)
                hsi = zoom(hsi, zoom_factors, order=1)

            # 保存为.mat文件
            output_file = os.path.join(output_dir, f"{set_name}_{video_dir}.mat")
            sio.savemat(output_file, {"img": hsi})

            print(f"已处理 {set_name}/{video_dir} -> {output_file}")

        return True

    except Exception as e:
        print(f"处理 {set_path} 时出错: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="预处理CAVE和KAIST数据集")
    parser.add_argument("--cave_dir", default="datasets/CAVE", help="CAVE数据集目录")
    parser.add_argument("--kaist_dir", default="datasets/KAIST", help="KAIST数据集目录")
    parser.add_argument("--output_dir", default="datasets", help="输出目录")
    args = parser.parse_args()

    # 创建输出目录
    cave_output = os.path.join(args.output_dir, "CAVE_processed")
    kaist_output = os.path.join(args.output_dir, "KAIST_processed")
    os.makedirs(cave_output, exist_ok=True)
    os.makedirs(kaist_output, exist_ok=True)

    print("开始处理CAVE数据集...")

    # 处理CAVE数据集
    cave_scenes = [
        d
        for d in os.listdir(args.cave_dir)
        if os.path.isdir(os.path.join(args.cave_dir, d))
    ]
    cave_scenes.sort()

    success_count = 0
    for i, scene in enumerate(cave_scenes):
        scene_path = os.path.join(args.cave_dir, scene)
        if process_cave_scene(scene_path, cave_output, i + 1):
            success_count += 1

    print(f"CAVE数据集处理完成: {success_count}/{len(cave_scenes)} 个场景处理成功")

    print("\n开始处理KAIST数据集...")

    # 处理KAIST数据集
    kaist_sets = [
        d
        for d in os.listdir(args.kaist_dir)
        if d.startswith("set") and os.path.isdir(os.path.join(args.kaist_dir, d))
    ]
    kaist_sets.sort()

    success_count = 0
    for set_name in kaist_sets:
        set_path = os.path.join(args.kaist_dir, set_name)
        if process_kaist_set(set_path, kaist_output, set_name):
            success_count += 1

    print(f"KAIST数据集处理完成: {success_count}/{len(kaist_sets)} 个set处理成功")

    print("\n数据预处理完成！")
    print(f"CAVE处理后的数据保存在: {cave_output}")
    print(f"KAIST处理后的数据保存在: {kaist_output}")


if __name__ == "__main__":
    main()
