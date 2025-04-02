#!/bin/bash

# LAW環境をアクティベート
source /home/shunya.seiya/miniconda3/etc/profile.d/conda.sh
conda activate law

# 必要なパッケージをインストール
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install timm

# mmdetection3dをインストール
cd /home/shunya.seiya/mlops_ws
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout -f v0.17.1
python setup.py develop

# NuScenes DevKitをインストール
pip install nuscenes-devkit==1.1.9
pip install yapf==0.40.1

echo "依存関係のインストールが完了しました。" 