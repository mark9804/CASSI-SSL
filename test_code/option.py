import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='gap_net',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/gap_net/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='gap_net', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='Y',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi_PhiPhiT',
                    help='the input mask of the network: Phi, Phi_PhiPhiT or None')

opt = parser.parse_args()
template.set_template(opt)

opt.mask_path = "../../../dataset/masks/simulation"
opt.test_path = "../../../dataset/KAIST/test/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False
