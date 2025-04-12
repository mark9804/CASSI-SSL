from architecture import *
from utils import *
import scipy.io as scio
import torch
import torch.nn.functional as F
import os
import numpy as np
from option import opt
from loss import tv, nuc_loss_v2
from tqdm import tqdm
from ssim_torch import ssim

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# Intialize mask
mask3d_batch, input_mask = init_mask(opt.mask_path, opt.input_mask, 1)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

def forward_model(x, Phi, step):
    nC = Phi.shape[1]
    mea = torch.sum(shift(x, step) * Phi, dim=1) / nC * 2
    return mea

i = 0
def test(model):
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.cuda().float()
    test_gt = test_gt[i:i+1, :, :, :]
    input_meas = init_meas(test_gt, mask3d_batch.unsqueeze(0), opt.input_setting)
    print(input_meas.max())
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)
    psnr_max = 0
    for i in tqdm(range(400)):
        optimizer.zero_grad()
        x1 = model(input_meas, input_mask)
        y_pred = forward_model(x1, input_mask[0], step=2)
        loss = torch.sqrt(F.mse_loss(y_pred, input_meas / 28 * 2)) + 0.0012 * tv(x1) + 0.0001 * nuc_loss_v2(x1, 24)

        loss.backward()
        optimizer.step()
        scheduler.step()

        pred = x1.squeeze(0)
        truth = test_gt.squeeze(0)

        print(torch_psnr(pred, truth).item(), ssim(pred.unsqueeze(0), truth.unsqueeze(0)).item())

    return x1, truth

def main():
    # model
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()
    pred, truth = test(model)
    name = opt.outf + 'Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})

if __name__ == '__main__':
    main()