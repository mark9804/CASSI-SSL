# -*- coding: utf-8 -*-
"""
CASSI-SSL 工具函数模块

本模块包含了CASSI-SSL框架中的核心工具函数，主要包括：
1. CASSI物理模型的实现（前向和逆向操作）
2. 数据加载和预处理功能
3. 掩模生成和初始化
4. 性能评估指标
5. 数据增强和批处理工具

这些工具函数是整个自监督学习框架的基础设施。
"""

import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from ssim_torch import ssim
import torchvision.transforms.functional as FF


def generate_masks(mask_path, batch_size):
    """
    生成基础的3D编码孔径掩模

    Args:
        mask_path (str): 掩模文件路径
        batch_size (int): 批处理大小

    Returns:
        torch.Tensor: 3D掩模张量 [B, C, H, W]

    说明：
        CASSI系统使用编码孔径来实现空间-光谱的联合编码。
        这个函数加载预设计的2D掩模并扩展到所有光谱通道。
    """
    mask = sio.loadmat(mask_path + "/mask.mat")
    mask = mask["mask"]

    # 将2D掩模扩展到28个光谱通道
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))

    # 调整维度顺序：[H, W, C] -> [C, H, W]
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)

    [nC, H, W] = mask3d.shape

    # 扩展到批处理维度
    if batch_size > 1:
        mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    else:
        mask3d_batch = mask3d.unsqueeze(0).cuda().float()
    return mask3d_batch


def generate_shift_masks(mask_path, batch_size):
    """
    生成CASSI系统所需的位移掩模和归一化因子

    Args:
        mask_path (str): 掩模文件路径
        batch_size (int): 批处理大小

    Returns:
        tuple: (Phi_batch, Phi_s_batch)
            - Phi_batch: 位移后的编码掩模 [B, C, H, W_shifted]
            - Phi_s_batch: 掩模的平方和，用于GAP算法归一化 [B, H, W_shifted]

    说明：
        这个函数实现了CASSI前向模型中的关键组件：
        1. 对每个光谱通道应用空间位移（模拟色散）
        2. 计算归一化因子Phi_s，用于GAP算法的数据一致性投影
    """
    mask = sio.loadmat(mask_path + "/mask.mat")
    mask = mask["mask"]

    # 生成3D掩模
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)

    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()

    # 应用空间位移，模拟CASSI系统的色散效应
    Phi_batch = shift(mask3d_batch)

    # 计算Phi的平方和，用于GAP算法中的归一化
    # 这是为了避免除零并确保数值稳定性
    Phi_s_batch = torch.sum(Phi_batch**2, 1)
    Phi_s_batch[Phi_s_batch == 0] = 1  # 防止除零

    return Phi_batch, Phi_s_batch


def LoadTraining(path):
    """
    加载训练数据集（CAVE数据集）

    Args:
        path (str): 数据集路径

    Returns:
        list: 包含所有训练高光谱图像的列表

    说明：
        CAVE数据集是一个标准的高光谱图像数据集，包含32个室内场景，
        每个场景有31个光谱通道。这里只使用前28个通道以匹配CASSI系统。
    """
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print("training sences:", len(scene_list))

    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        scene_num = int(scene_list[i].split(".")[0][5:])

        # 只使用场景编号<=205的数据作为训练集
        if scene_num <= 205:
            if "mat" not in scene_path:
                continue

            img_dict = sio.loadmat(scene_path)

            # 处理不同的数据格式
            if "img_expand" in img_dict:
                img = img_dict["img_expand"] / 65536.0  # 归一化到[0,1]
            elif "img" in img_dict:
                img = img_dict["img"] / 65536.0

            img = img.astype(np.float32)
            imgs.append(img)
            print("Sence {} is loaded. {}".format(i, scene_list[i]))

    return imgs


def LoadTest(path_test):
    """
    加载测试数据集（KAIST数据集）

    Args:
        path_test (str): 测试数据集路径

    Returns:
        torch.Tensor: 测试数据张量 [N, C, H, W]

    说明：
        KAIST数据集用于评估模型的重建性能，
        包含多个场景的高光谱图像。
    """
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))

    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)["img"]
        test_data[i, :, :, :] = img

    # 转换维度顺序：[N, H, W, C] -> [N, C, H, W]
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data


def LoadMeasurement(path_test_meas):
    """
    加载预计算的测量数据

    Args:
        path_test_meas (str): 测量数据文件路径

    Returns:
        torch.Tensor: 测量数据张量
    """
    img = sio.loadmat(path_test_meas)["simulation_test"]
    test_data = img
    test_data = torch.from_numpy(test_data)
    return test_data


def torch_psnr(img, ref):
    """
    计算峰值信噪比（PSNR）

    PSNR是评估图像重建质量的重要指标，定义为：
    PSNR = 10 * log10((MAX²) / MSE)
    其中MAX是图像的最大可能像素值，MSE是均方误差。

    Args:
        img (torch.Tensor): 重建图像 [C, H, W]
        ref (torch.Tensor): 参考图像 [C, H, W]

    Returns:
        torch.Tensor: 平均PSNR值

    说明：
        这里采用与DGSMP相同的计算方法，
        先对每个光谱通道分别计算PSNR，然后取平均。
    """
    # 将[0,1]范围的图像缩放到[0,255]并取整
    img = (img * 256).round()
    ref = (ref * 256).round()

    nC = img.shape[0]
    psnr = 0

    # 对每个光谱通道分别计算PSNR
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255 * 255) / mse)

    return psnr / nC


def torch_ssim(img, ref):
    """
    计算结构相似性指数（SSIM）

    SSIM是一种感知质量指标，考虑了亮度、对比度和结构信息。
    相比PSNR，SSIM更符合人类视觉感知。

    Args:
        img (torch.Tensor): 重建图像 [C, H, W]
        ref (torch.Tensor): 参考图像 [C, H, W]

    Returns:
        torch.Tensor: SSIM值
    """
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


def time2file_name(time):
    """
    将时间字符串转换为文件名格式

    Args:
        time (str): 时间字符串

    Returns:
        str: 格式化的文件名
    """
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = (
        year + "_" + month + "_" + day + "_" + hour + "_" + minute + "_" + second
    )
    return time_filename


def shuffle_crop(train_data, batch_size, crop_size=256, argument=True):
    """
    随机裁剪和数据增强函数

    这个函数实现了一种混合的数据增强策略：
    1. 一半数据使用原始图像的随机裁剪
    2. 另一半数据使用拼接增强（4个小块拼成一个大图）

    Args:
        train_data (list): 训练数据列表
        batch_size (int): 批处理大小
        crop_size (int): 裁剪尺寸，默认256
        argument (bool): 是否启用数据增强

    Returns:
        torch.Tensor: 处理后的批次数据 [B, C, H, W]

    数据增强的作用：
        1. 增加训练数据的多样性
        2. 提高模型的泛化能力
        3. 模拟不同的空间结构和光谱分布
    """
    if argument:
        gt_batch = []

        # 前一半数据：使用原始数据的随机裁剪
        index = np.random.choice(range(len(train_data)), batch_size // 2)
        processed_data = np.zeros(
            (batch_size // 2, crop_size, crop_size, 28), dtype=np.float32
        )

        for i in range(batch_size // 2):
            img = train_data[index[i]]
            h, w, _ = img.shape

            # 随机选择裁剪位置
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[
                x_index : x_index + crop_size, y_index : y_index + crop_size, :
            ]

        processed_data = (
            torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        )

        # 对每个样本应用随机变换（旋转、翻转等）
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))

        # 后一半数据：使用拼接增强
        processed_data = np.zeros((4, 128, 128, 28), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            # 从4个不同场景中随机选择4个小块
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h - crop_size // 2)
                y_index = np.random.randint(0, w - crop_size // 2)
                processed_data[j] = train_data[sample_list[j]][
                    x_index : x_index + crop_size // 2,
                    y_index : y_index + crop_size // 2,
                    :,
                ]

            gt_batch_2 = torch.from_numpy(
                np.transpose(processed_data, (0, 3, 1, 2))
            ).cuda()
            gt_batch.append(arguement_2(gt_batch_2))

        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        # 简单的随机裁剪，不使用数据增强
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros(
            (batch_size, crop_size, crop_size, 28), dtype=np.float32
        )

        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][
                x_index : x_index + crop_size, y_index : y_index + crop_size, :
            ]

        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch


def arguement_1(x):
    """
    单个图像的随机几何变换

    Args:
        x (torch.Tensor): 输入图像 [C, H, W]

    Returns:
        torch.Tensor: 变换后的图像 [C, H, W]

    变换类型：
        1. 随机旋转（0°, 90°, 180°, 270°）
        2. 随机垂直翻转
        3. 随机水平翻转
    """
    rotTimes = random.randint(0, 3)  # 旋转次数（每次90度）
    vFlip = random.randint(0, 1)  # 是否垂直翻转
    hFlip = random.randint(0, 1)  # 是否水平翻转

    # 随机旋转
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))

    # 随机垂直翻转
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))

    # 随机水平翻转
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))

    return x


def arguement_2(generate_gt):
    """
    四块拼接增强：将4个128×128的块拼接成一个256×256的图像

    Args:
        generate_gt (torch.Tensor): 4个小块 [4, C, 128, 128]

    Returns:
        torch.Tensor: 拼接后的大图像 [C, 256, 256]

    拼接方式：
        [块0] [块1]
        [块2] [块3]

    这种增强方式可以：
        1. 增加空间结构的多样性
        2. 模拟复杂场景的光谱分布
        3. 提高模型对不同区域特征的适应能力
    """
    c, h, w = generate_gt.shape[1], 256, 256
    divid_point_h = 128
    divid_point_w = 128

    output_img = torch.zeros(c, h, w).cuda()

    # 左上角
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    # 右上角
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    # 左下角
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    # 右下角
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]

    return output_img


def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    """
    生成CASSI压缩测量值

    This function implements the core part of the CASSI forward imaging model:
    Y = Σ[Shift(Mask ⊙ HSI, i)]

    Args:
        data_batch (torch.Tensor): Batch of hyperspectral images [B, C, H, W]
        mask3d_batch (torch.Tensor): 3D mask [B, C, H, W]
        Y2H (bool): Whether to convert from measurement to HSI format
        mul_mask (bool): Whether to multiply with mask

    Returns:
        torch.Tensor: Measurement or processed HSI

    Processing steps:
        1. Mask encoding: HSI ⊙ Mask
        2. Spatial shift: Simulate dispersion effect
        3. Spectral integration: Summing up in spectral dimension to get 2D measurement
        4. Optional reverse processing: Recover from measurement to HSI format
    """
    nC = data_batch.shape[1]

    # Steps 1&2: Apply mask and spatial shift
    temp = shift(mask3d_batch * data_batch, 2)

    # Step 3: Summing up in spectral dimension to get 2D measurement
    meas = torch.sum(temp, 1)

    if Y2H:
        # Normalization processing
        meas = meas / nC * 2

        # Recover from measurement to HSI format (for network input)
        H = shift_back(meas)

        if mul_mask:
            # Optional: Apply mask again
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H

    return meas


def shift(inputs, step=2):
    """
    CASSI forward spatial shift operation: Simulate prism dispersion effect

    In the CASSI system, light of different wavelengths after passing through the prism will produce different degrees of dispersion,
    causing different spectral channels to be shifted relative to each other in space.

    Args:
        inputs (torch.Tensor): Input HSI [B, C, H, W]
        step (int): Shift step, default 2 pixels

    Returns:
        torch.Tensor: Shifted HSI [B, C, H, W_extended]

    Shift rule:
        - 0th channel: No shift
        - ith channel: Shift right by step*i pixels
        - Output width: W + (C-1)*step
    """
    [bs, nC, row, col] = inputs.shape

    # Create extended output tensor to accommodate shift
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()

    # Apply corresponding shift to each spectral channel
    for i in range(nC):
        output[:, i, :, step * i : step * i + col] = inputs[:, i, :, :]

    return output


def shift_back(inputs, step=2):
    """
    CASSI reverse spatial shift operation: Recover HSI format from 2D measurement

    This is the reverse process of shift operation, rearranging 2D measurement values into 3D HSI format.
    Note: This is not a true inverse operation, but provides a suitable input format for the network.

    Args:
        inputs (torch.Tensor): 2D measurement values [B, H, W_extended]
        step (int): Shift step, default 2 pixels

    Returns:
        torch.Tensor: HSI format data [B, C, H, W]

    Note:
        This operation provides a suitable initial estimate for GAP-net,
        equivalent to the implementation of At(y) operation.
    """
    [bs, row, col] = inputs.shape
    nC = 28  # Fixed number of spectral channels

    # Calculate original image width
    output_width = col - (nC - 1) * step
    output = torch.zeros(bs, nC, row, output_width).cuda().float()

    # Extract each spectral channel from different positions
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i : step * i + output_width]

    return output


def gen_log(model_path):
    """
    Generate logger

    Args:
        model_path (str): Model save path

    Returns:
        logging.Logger: Configured logger

    Function:
        1. Output to both file and console
        2. Record important information during training
        3. Facilitate tracking and analysis of experimental results
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    # File handler: Write log to file
    log_file = model_path + "/log.txt"
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Console handler: Output to terminal
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def init_mask(mask_path, mask_type, batch_size):
    """
    Initialize different types of mask

    Args:
        mask_path (str): Mask file path
        mask_type (str): Mask type
            - 'Phi': Shifted mask
            - 'Phi_PhiPhiT': Mask pair required for GAP algorithm
            - 'Mask': Original 3D mask
            - None: Do not use mask
        batch_size (int): Batch size

    Returns:
        tuple: (mask3d_batch, input_mask)
            - mask3d_batch: Basic 3D mask
            - input_mask: Mask format required for network input
    """
    mask3d_batch = generate_masks(mask_path, batch_size)

    if mask_type == "Phi":
        # Use only shifted mask
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == "Phi_PhiPhiT":
        # Mask pair required for GAP algorithm: Phi and Phi_s
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == "Mask":
        # Original 3D mask
        input_mask = mask3d_batch
    elif mask_type == None:
        # Do not use mask (usually not happen)
        input_mask = None

    return mask3d_batch, input_mask


def init_meas(gt, mask, input_setting):
    """
    Generate different format network input based on setting

    Args:
        gt (torch.Tensor): True hyperspectral image
        mask (torch.Tensor): Mask
        input_setting (str): Input setting
            - 'H': HSI format recovered from measurement
            - 'HM': H multiplied by mask
            - 'Y': Original 2D measurement

    Returns:
        torch.Tensor: Network input data

    Note:
        Different input formats are suitable for different network architectures and training strategies.
        'Y' format is closer to the true CASSI measurement.
    """
    if input_setting == "H":
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == "HM":
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == "Y":
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)

    return input_meas


def checkpoint(model, epoch, model_path, logger):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        epoch (int): Current training round
        model_path (str): Model save path
        logger: Logger

    Function:
        Save trained model parameters for subsequent testing and fine-tuning.
    """
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))
