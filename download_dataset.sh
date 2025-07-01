#!/bin/bash

if [ ! -d "datasets" ]; then
    mkdir datasets
fi

# CAVE
# https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip
curl -L -o datasets/complete_ms_data.zip https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip
unzip datasets/complete_ms_data.zip -d datasets/CAVE
rm datasets/complete_ms_data.zip

# KAIST
# https://www.kaggle.com/api/v1/datasets/download/adlteam/kaist-dataset
curl -L -o datasets/kaist-dataset.zip https://www.kaggle.com/api/v1/datasets/download/adlteam/kaist-dataset
unzip datasets/kaist-dataset.zip -d datasets/KAIST
rm datasets/kaist-dataset.zip

# download masks

if [ ! -d "datasets/masks" ]; then
    mkdir datasets/masks
fi

if [ ! -d "datasets/masks/real" ]; then
    mkdir datasets/masks/real
fi

curl -L -o datasets/masks/real/mask.mat https://github.com/mengziyi64/TSA-Net/raw/refs/heads/master/TSA_Net_realdata/Data/mask.mat
curl -L -o datasets/masks/real/mask_3d_shift_uint16.mat https://github.com/mengziyi64/TSA-Net/raw/refs/heads/master/TSA_Net_realdata/Data/mask_3d_shift_uint16.mat