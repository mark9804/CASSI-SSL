# -*- coding: utf-8 -*-
"""
CASSI-SSL 自监督损失函数模块

本模块实现了论文中提到的关键损失函数，特别是谱低秩损失（nuc_loss_v2）和总变分损失（tv）。
这些损失函数是实现自监督学习的核心，它们将高光谱图像的物理先验转化为可微的约束，
在没有真实标签的情况下引导网络学习正确的信号结构。

核心损失函数：
1. 总变分损失（TV Loss）: 促进空间平滑性，减少噪声和伪影
2. 谱低秩损失（Nuclear Norm Loss）: 强制光谱的低秩结构，这是论文的核心创新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import image_gradients as ig_


def tv(x: torch.tensor):
    """
    总变分损失（Total Variation Loss）

    总变分正则化是一种经典的图像处理技术，基于自然图像通常在局部区域内平滑的先验知识。
    通过最小化图像的总变分，可以促进空间平滑性，减少噪声和伪影，同时保留重要的边缘信息。

    数学定义：
        TV(x) = Σ_k [|∇_h x_k|_1 + |∇_v x_k|_1]
        其中 ∇_h 和 ∇_v 分别是水平和垂直方向的梯度算子，k 是光谱通道索引

    Args:
        x (torch.Tensor): 输入的高光谱图像 [B, C, H, W]

    Returns:
        torch.Tensor: 总变分损失值（标量）

    物理意义：
        - 自然场景中的大部分区域在空间上是平滑的
        - 通过惩罚大的梯度值来减少噪声
        - L1范数比L2范数更好地保留边缘信息
    """
    # 计算水平和垂直方向的梯度
    dx, dy = ig_(x)

    # 计算梯度的L1范数并求平均
    # L1范数比L2范数更适合保持边缘，因为它对大梯度的惩罚较小
    return dx.abs().mean() + dy.abs().mean()


def nuc_loss_v2(x: torch.tensor, patch_size, eps=1e-4):
    """
    谱低秩损失（Spectral Low-Rank Loss）- 论文的核心创新

    这是论文最重要的贡献之一。该损失函数将高光谱图像固有的光谱低秩特性转化为
    可微的数学约束，使得网络能够在没有真实标签的情况下学习正确的光谱结构。

    核心思想：
        自然场景中，一个局部区域内不同像素的光谱曲线通常是高度相关的，
        可以由少数几个基本光谱线性组合而成。这意味着将这些光谱向量排列成
        矩阵时，该矩阵应该具有低秩特性（大部分奇异值接近零）。

    实现步骤：
        1. 分块：将3D高光谱图像分割成重叠的3D块
        2. 展开：将每个3D块重新排列为2D矩阵（像素×光谱）
        3. SVD：计算每个矩阵的奇异值分解
        4. 损失：对所有奇异值的对数求和并平均

    数学表达：
        L_lr = (1/N_p) Σ_i Σ_r log(σ_r(Z_i) + ε)
        其中 σ_r(Z_i) 是第i个块矩阵的第r个奇异值

    Args:
        x (torch.Tensor): 输入的高光谱图像 [B, C, H, W]
        B: batch size, 批大小
        C: channel, 光谱通道数
        H: 高度
        W: 宽度
        patch_size (int): 分块的空间尺寸，通常为8或16
        eps (float): 防止log(0)的小常数

    Returns:
        torch.Tensor: 谱低秩损失值（标量）

    为什么使用对数？
        1. 对数函数对较大的奇异值（主要信息）惩罚较小
        2. 对较小的奇异值（噪声/冗余）惩罚较大
        3. 这种加权策略有助于保留主要光谱信息同时抑制噪声
    """
    B, C, H, W = x.shape  # 获取张量形状

    # 设置分块的步长，通常为块大小的一半，产生重叠块
    # 重叠块可以提供更好的空间一致性
    stride = int(patch_size / 2 - 1)

    # 使用nn.Unfold高效地提取重叠的空间块
    # unfold操作将输入张量分解为滑动的局部块
    unfold = nn.Unfold(kernel_size=patch_size, stride=stride)

    # 展开操作：[B, C, H, W] -> [B, C*patch_size*patch_size, num_patches]
    x = unfold(x)

    # 重新排列维度：[B, num_patches, C*patch_size*patch_size]
    x = x.permute(0, 2, 1)

    # 重塑为所需的矩阵形式：[B, num_patches, C, patch_size*patch_size]
    # 这样每个 [C, patch_size*patch_size] 矩阵表示一个空间块的所有光谱信息
    x = x.reshape(B, -1, C, patch_size * patch_size)

    # 计算奇异值分解（SVD）
    # torch.linalg.svdvals 只计算奇异值，比完整SVD更高效
    # S的形状：[B*num_patches, min(C, patch_size*patch_size)]
    S = torch.linalg.svdvals(x)

    # 计算核范数的对数形式
    # 加上eps防止log(0)导致的数值不稳定
    nuc_norm_log = torch.mean(torch.log(S + eps))

    return nuc_norm_log
