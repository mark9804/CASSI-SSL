# -*- coding: utf-8 -*-
"""
CASSI-SSL 自监督高光谱图像重建训练脚本

本脚本实现了基于谱低秩先验的自监督学习框架中的第一阶段训练（SST - Self-Supervised Training）。
主要目的是训练一个通用的GAP-net模型，使其能够从二维压缩测量值重建三维高光谱图像，
而无需真实的高光谱图像作为监督标签。

核心创新点：
1. 使用三元复合损失函数（测量一致性损失 + 总变分损失 + 谱低秩损失）
2. 通过物理先验约束引导网络学习正确的信号结构
3. 仅使用测量值训练，完全摆脱对真实HSI标签的依赖
"""

from architecture import *
from utils import *
import torch
import os
from option import opt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
import scipy.io as scio
import time
import numpy as np
from torch.autograd import Variable
import datetime
import torch.nn.functional as F
from loss import tv, nuc_loss_v2


def set_seed(seed):
    """
    设置随机种子，确保实验的可重复性

    Args:
        seed (int): 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def forward_model(x, Phi, step):
    """
    CASSI前向成像模型的实现

    根据论文公式，CASSI系统将3D高光谱图像压缩为2D测量值：
    Y = Φ(X) = Σ[Coded_Aperture ⊙ Shift(X, i)]

    Args:
        x (torch.Tensor): 输入的高光谱图像 [B, C, H, W]
        Phi (torch.Tensor): 编码孔径掩模 [B, C, H, W]
        step (int): 色散位移步长，通常为2

    Returns:
        torch.Tensor: 2D压缩测量值 [B, H, W_shifted]

    原理：
    1. 对高光谱图像的每个波段进行空间位移（模拟色散）
    2. 与编码孔径掩模相乘（模拟编码）
    3. 在光谱维度求和得到最终的2D测量值
    """
    nC = Phi.shape[1]  # 光谱通道数
    # 应用空间位移和掩模编码，然后在光谱维度求和
    mea = torch.sum(shift(x, step) * Phi, dim=1) / nC * 2
    return mea


# 设置随机种子确保可重复性
set_seed(3407)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception("NO GPU!")

# 初始化CASSI系统掩模
# mask3d_batch_train: 用于训练的3D掩模 [B, C, H, W]
# input_mask_train: GAP-net所需的输入掩模格式，包含Phi和Phi_s
mask3d_batch_train, input_mask_train = init_mask(
    opt.mask_path, opt.input_mask, opt.batch_size
)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)

# 数据集加载
# train_set: CAVE数据集的训练高光谱图像列表
# test_data: KAIST数据集的测试高光谱图像
train_set = LoadTraining(opt.data_path)
test_data = LoadTest(opt.test_path)

# 创建保存路径
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + "/result/"
model_path = opt.outf + date_time + "/model/"
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# 初始化GAP-net模型
model = model_generator(opt.method, opt.pretrained_model_path).cuda()

# 优化器设置：使用Adam优化器和余弦退火学习率调度
optimizer = torch.optim.Adam(
    model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999)
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, opt.max_epoch, eta_min=1e-6
)
mse = torch.nn.MSELoss().cuda()


def train(epoch, logger):
    """
    单个epoch的训练函数

    实现自监督训练的核心逻辑：
    1. 从训练数据中生成压缩测量值
    2. 通过GAP-net重建高光谱图像
    3. 使用三元复合损失函数进行优化

    Args:
        epoch (int): 当前训练轮数
        logger: 日志记录器

    损失函数构成：
    - 测量一致性损失（L_mc）: 确保重建结果能重现输入测量值
    - 总变分损失（L_tv）: 促进空间平滑性
    - 谱低秩损失（L_lr）: 强制光谱低秩结构（仅在epoch>60后启用）
    """
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))

    for i in range(batch_num):
        # 随机裁剪并生成训练批次
        gt_batch = shuffle_crop(train_set, opt.batch_size)
        gt = Variable(gt_batch).cuda().float()

        # 生成压缩测量值（模拟CASSI前向过程）
        input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)

        optimizer.zero_grad()

        # GAP-net前向传播：从测量值重建高光谱图像
        model_out = model(input_meas, input_mask_train)

        # 分阶段损失函数策略
        if epoch <= 60:
            # 前60个epoch：仅使用测量一致性损失和总变分损失
            # 这是为了让网络首先学会基本的重建能力
            loss = torch.sqrt(
                F.mse_loss(
                    forward_model(model_out, input_mask_train[0], 2),
                    input_meas / 28 * 2,
                )
            ) + 0.01 * tv(model_out)
        else:
            # 60个epoch后：加入谱低秩损失
            # 这是论文的核心创新，强制网络学习光谱的低秩结构
            loss = (
                torch.sqrt(
                    F.mse_loss(
                        forward_model(model_out, input_mask_train[0], 2),
                        input_meas / 28 * 2,
                    )
                )
                + 0.001 * tv(model_out)
                + 0.0001 * nuc_loss_v2(model_out, 16)
            )

        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

    end = time.time()
    logger.info(
        "===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".format(
            epoch, epoch_loss / batch_num, (end - begin)
        )
    )
    return 0


def test(epoch, logger):
    """
    测试函数：评估当前模型在测试集上的性能

    Args:
        epoch (int): 当前训练轮数
        logger: 日志记录器

    Returns:
        tuple: 包含重建结果、真实值和性能指标的元组

    评估指标：
    - PSNR (Peak Signal-to-Noise Ratio): 峰值信噪比
    - SSIM (Structural Similarity Index): 结构相似性指数
    """
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()

    # 为测试数据生成压缩测量值
    input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)

    model.eval()
    begin = time.time()
    with torch.no_grad():
        # 网络推理：从测量值重建高光谱图像
        model_out = model(input_meas, input_mask_test)

    end = time.time()

    # 计算每个测试样本的PSNR和SSIM
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())

    # 数据格式转换：从[B,C,H,W]转为[B,H,W,C]
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(
        np.float32
    )
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)

    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))

    logger.info(
        "===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}".format(
            epoch, psnr_mean, ssim_mean, (end - begin)
        )
    )

    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean


def main():
    """
    主训练循环

    实现完整的自监督训练流程：
    1. 初始化日志系统
    2. 执行多个epoch的训练和测试
    3. 保存最佳性能的模型和结果

    这个阶段训练得到的模型将作为SSTT（自监督训练和微调）阶段的预训练模型。
    """
    logger = gen_log(model_path)
    logger.info(
        "Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size)
    )

    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        # 训练一个epoch
        train(epoch, logger)

        # 在测试集上评估性能
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)

        # 学习率调度
        scheduler.step()

        # 保存最佳模型
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            # 只保存PSNR超过25dB的模型（表明模型已有良好性能）
            if psnr_mean > 25:
                name = (
                    result_path
                    + "/"
                    + "Test_{}_{:.2f}_{:.3f}".format(epoch, psnr_max, ssim_mean)
                    + ".mat"
                )
                scio.savemat(
                    name,
                    {
                        "truth": truth,
                        "pred": pred,
                        "psnr_list": psnr_all,
                        "ssim_list": ssim_all,
                    },
                )
                checkpoint(model, epoch, model_path, logger)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()
