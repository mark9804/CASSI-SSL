from architecture import *
from utils import *
import torch
import os
from option import opt
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
import scipy.io as scio
import time
import numpy as np
from torch.autograd import Variable
import datetime
import torch.nn.functional as F
from loss import tv, nuc_loss_v2

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def forward_model(x, Phi, step):
    nC = Phi.shape[1]
    mea = torch.sum(shift(x, step) * Phi, dim=1) / nC * 2
    return mea

set_seed(3407)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# init mask
mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)

# dataset
train_set = LoadTraining(opt.data_path)
test_data = LoadTest(opt.test_path)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

model = model_generator(opt.method, opt.pretrained_model_path).cuda()

# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
mse = torch.nn.MSELoss().cuda()

def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    for i in range(batch_num):
        gt_batch = shuffle_crop(train_set, opt.batch_size)
        gt = Variable(gt_batch).cuda().float()
        input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)
        optimizer.zero_grad()

        model_out = model(input_meas, input_mask_train)
        if epoch <= 60:
            loss = torch.sqrt(F.mse_loss(forward_model(model_out, input_mask_train[0], 2), input_meas / 28 * 2)) + 0.01 * tv(model_out)
        else:
            loss = torch.sqrt(F.mse_loss(forward_model(model_out, input_mask_train[0], 2), input_meas / 28 * 2)) + 0.001 * tv(model_out) + 0.0001 * nuc_loss_v2(model_out, 16)

        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    end = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, epoch_loss / batch_num, (end - begin)))
    return 0

def test(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
    model.eval()
    begin = time.time()
    with torch.no_grad():
        model_out = model(input_meas, input_mask_test)

    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)
        scheduler.step()
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 25:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                checkpoint(model, epoch, model_path, logger)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


