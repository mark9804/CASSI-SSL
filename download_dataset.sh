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