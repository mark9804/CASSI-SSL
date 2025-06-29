#!/bin/bash

# create target dir if not exists
if [ ! -d "datasets" ]; then
    mkdir datasets
fi

# download CAVE and unzip
# https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip
curl -L -o datasets/complete_ms_data.zip https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip

unzip datasets/complete_ms_data.zip -d datasets/CAVE

# remove zip
rm datasets/complete_ms_data.zip

# download KAIST and untar
# https://www.kaggle.com/api/v1/datasets/download/adlteam/kaist-dataset

curl -L -o datasets/kaist-dataset.tar.gz https://www.kaggle.com/api/v1/datasets/download/adlteam/kaist-dataset

tar -xzf datasets/kaist-dataset.tar.gz -C datasets/KAIST

# remove tar
rm datasets/kaist-dataset.tar.gz