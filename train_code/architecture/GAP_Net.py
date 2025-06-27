# -*- coding: utf-8 -*-
"""
GAP-Net: 基于广义交替投影的深度展开网络

GAP-Net是一个深度展开网络（Deep Unfolding Network），将经典的GAP优化算法的迭代过程
展开成神经网络的前向传播过程。每个网络阶段对应GAP算法的一次迭代，包含两个核心步骤：
1. 数据一致性投影：确保解满足物理约束 y = Φx
2. 去噪步骤：通过U-Net去噪器施加图像先验

这种设计兼具了优化算法的可解释性和深度学习的表达能力。
"""

import torch.nn.functional as F
import torch
import torch.nn as nn


def A(x, Phi):
    """
    CASSI前向算子：模拟从高光谱图像到压缩测量的过程

    Args:
        x (torch.Tensor): 高光谱图像 [B, C, H, W]
        Phi (torch.Tensor): 编码孔径掩模 [B, C, H, W]

    Returns:
        torch.Tensor: 压缩测量值 [B, H, W]

    原理：Φ ⊙ X 然后在光谱维度求和
    """
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y


def At(y, Phi):
    """
    CASSI转置算子：从压缩测量到高光谱图像的初始估计

    Args:
        y (torch.Tensor): 压缩测量值 [B, H, W]
        Phi (torch.Tensor): 编码孔径掩模 [B, C, H, W]

    Returns:
        torch.Tensor: 高光谱图像的初始估计 [B, C, H, W]

    原理：将测量值y复制到所有光谱通道，然后与掩模相乘
    这是最小二乘意义下的初始解
    """
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    """
    3D空间位移操作：模拟CASSI系统中的色散效应

    Args:
        inputs (torch.Tensor): 输入高光谱图像 [B, C, H, W]
        step (int): 位移步长，默认为2像素

    Returns:
        torch.Tensor: 位移后的高光谱图像

    原理：不同光谱通道在水平方向有不同的位移量
    第i个通道位移 step*i 个像素，模拟棱镜色散效应
    """
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=step * i, dims=2)
    return inputs


def shift_back_3d(inputs, step=2):
    """
    3D空间位移的逆操作：恢复原始的光谱对齐

    Args:
        inputs (torch.Tensor): 位移后的高光谱图像 [B, C, H, W]
        step (int): 位移步长，默认为2像素

    Returns:
        torch.Tensor: 恢复对齐的高光谱图像

    原理：与shift_3d相反，将位移的光谱通道恢复到原始位置
    """
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(
            inputs[:, i, :, :], shifts=(-1) * step * i, dims=2
        )
    return inputs


class double_conv(nn.Module):
    """
    双卷积块：U-Net的基本构建单元

    包含两个连续的3x3卷积层，每个卷积后跟ReLU激活函数
    这是U-Net中的标准卷积块，用于特征提取和维度变换
    """

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x


class Unet(nn.Module):
    """
    U-Net去噪器：GAP-Net中的核心去噪模块

    U-Net是一个编码器-解码器架构，具有跳跃连接：
    1. 编码器：通过下采样提取多尺度特征
    2. 解码器：通过上采样恢复空间分辨率
    3. 跳跃连接：保留细节信息
    4. 残差连接：输出 = 输入 + 网络预测的残差

    在GAP-Net中，U-Net充当了一个强大的图像先验，
    替代了传统优化算法中简单的手工先验（如总变分）
    """

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        # 编码器路径：逐步下采样提取特征
        self.dconv_down1 = double_conv(in_ch, 32)  # 第一层：输入通道 -> 32通道
        self.dconv_down2 = double_conv(32, 64)  # 第二层：32 -> 64通道
        self.dconv_down3 = double_conv(64, 128)  # 第三层：64 -> 128通道

        self.maxpool = nn.MaxPool2d(2)  # 2x2最大池化用于下采样

        # 解码器路径：逐步上采样恢复分辨率
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 转置卷积上采样
            nn.ReLU(inplace=True),
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(inplace=True)
        )

        # 融合跳跃连接的卷积层
        self.dconv_up2 = double_conv(64 + 64, 64)  # 64(上采样) + 64(跳跃) = 128输入
        self.dconv_up1 = double_conv(32 + 32, 32)  # 32(上采样) + 32(跳跃) = 64输入

        # 输出层：生成最终的去噪结果
        self.conv_last = nn.Conv2d(32, out_ch, 1)  # 1x1卷积调整通道数
        self.afn_last = nn.Tanh()  # Tanh激活函数限制输出范围

    def forward(self, x):
        """
        U-Net前向传播

        Implement the forward pass of U-Net, which includes encoding and decoding
        """
        b, c, h_inp, w_inp = x.shape

        # 为了适应下采样操作，需要对输入进行填充
        # 确保高宽都是8的倍数（因为有3次下采样，2^3=8）
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
        inputs = x  # 保存输入用于残差连接

        # 编码器路径：提取多尺度特征
        conv1 = self.dconv_down1(x)  # 第一层特征
        x = self.maxpool(conv1)  # 下采样
        conv2 = self.dconv_down2(x)  # 第二层特征
        x = self.maxpool(conv2)  # 下采样
        conv3 = self.dconv_down3(x)  # 第三层特征（瓶颈层）

        # 解码器路径：恢复空间分辨率并融合特征
        x = self.upsample2(conv3)  # 上采样
        x = torch.cat([x, conv2], dim=1)  # 跳跃连接
        x = self.dconv_up2(x)  # 融合特征

        x = self.upsample1(x)  # 上采样
        x = torch.cat([x, conv1], dim=1)  # 跳跃连接
        x = self.dconv_up1(x)  # 融合特征

        # 输出层：生成残差并与输入相加
        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs  # 残差连接：输出 = 输入 + 残差

        # 裁剪到原始输入尺寸
        return out[:, :, :h_inp, :w_inp]


class DoubleConv(nn.Module):
    """
    带批归一化的双卷积块（未在主模型中使用）

    这是一个更标准的双卷积实现，包含批归一化层
    可以提供更稳定的训练过程
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class GAP_net(nn.Module):
    """
    GAP-Net主网络：基于广义交替投影的深度展开网络

    GAP-Net将GAP优化算法展开为神经网络，每个stage对应一次GAP迭代：

    GAP算法解决的优化问题：
        min_{x,v} (1/2)||x-v||²₂ + R(v)  s.t. y = Φx

    GAP迭代步骤：
        1. x^{k+1} = v^k + Φᵀ(y - Φv^k)     [数据一致性投影]
        2. v^{k+1} = D(x^{k+1})              [去噪步骤]

    网络参数：
        - stage: GAP迭代的阶段数，更多阶段通常带来更好的重建质量
        - ch: 光谱通道数，对于高光谱图像通常是28或31
        - step: 色散位移步长，模拟CASSI系统的物理参数
    """

    def __init__(self, stage=9, ch=28, step=2):
        super(GAP_net, self).__init__()

        self.stage = stage
        self.ch = ch
        self.step = step

        # 为每个GAP阶段创建一个独立的U-Net去噪器
        # 这允许不同阶段学习不同的去噪策略
        self.denoisers = nn.ModuleList([])
        for i in range(stage):
            self.denoisers.append(Unet(ch, ch))

    def forward(self, y, input_mask=None):
        """
        GAP-Net前向传播：实现多阶段的GAP迭代重建

        Args:
            y (torch.Tensor): 压缩测量值 [B, H, W]
            input_mask: 包含Phi和Phi_s的元组，用于GAP算法
                       Phi: 编码孔径掩模 [B, C, H, W]
                       Phi_s: Phi的平方和，用于归一化 [B, H, W]

        Returns:
            torch.Tensor: 重建的高光谱图像 [B, C, H, W_out]

        网络流程：
            1. 初始化：x⁰ = Φᵀy
            2. 对于每个阶段k=1,2,...,stage：
               a) 数据一致性投影：x = x + Φᵀ((y-Φx)/Φₛ)
               b) 空间位移逆变换：恢复光谱对齐
               c) U-Net去噪：v = D(x)
               d) 空间位移变换：为下一阶段准备（除最后阶段）
            3. 输出裁剪：移除色散造成的额外宽度
        """
        B, H, W = y.shape

        # 如果没有提供掩模，则生成随机掩模（通常不会发生）
        if input_mask == None:
            Phi = torch.rand((B, self.ch, H, W)).to(y.device)
            Phi_s = torch.rand((B, H, W)).to(y.device)
        else:
            Phi, Phi_s = input_mask

        x_list = []  # 用于存储中间结果（如需要）

        # GAP算法初始化：x⁰ = Φᵀy
        x = At(y, Phi)  # 转置算子提供初始估计

        # GAP迭代：每个阶段对应一次完整的GAP迭代
        for i, denoiser in enumerate(self.denoisers):
            # 步骤1：数据一致性投影
            # 计算当前估计的前向投影
            yb = A(x, Phi)

            # 根据测量残差更新估计：x = x + Φᵀ((y-yb)/Φₛ)
            # 这确保了重建结果在数据一致性约束下的最优性
            x = x + At(torch.div(y - yb, Phi_s), Phi)

            # 步骤2：去噪处理
            # 首先恢复光谱对齐（逆色散变换）
            x = shift_back_3d(x, self.step)

            # 使用U-Net去噪器施加图像先验
            # 这是深度展开的核心：用学习的先验替代手工先验
            x = denoiser(x)

            # 如果不是最后一个阶段，进行色散变换为下一阶段准备
            if i == self.stage - 1:
                break  # 最后阶段不需要色散变换
            else:
                x = shift_3d(x, self.step)

        # 输出裁剪：移除由于色散造成的额外宽度
        # 原始宽度 + (ch-1)*step 减少到原始宽度
        return x[:, :, :, 0 : (W - (self.ch - 1) * self.step)]
