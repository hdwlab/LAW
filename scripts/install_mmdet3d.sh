#!/bin/bash

# LAW環境をアクティベート
echo "Conda環境を有効化します..."
source /home/shunya.seiya/miniconda3/etc/profile.d/conda.sh
conda activate law
echo "Conda環境の有効化完了"

# mmsegmentationをインストール
echo "mmsegmentationをインストールします..."
pip install mmsegmentation==0.14.1

# LAWディレクトリに移動し、projects/mmdet3d_pluginディレクトリをPYTHONPATHに追加
cd /home/shunya.seiya/mlops_ws/LAW
export PYTHONPATH=$PYTHONPATH:/home/shunya.seiya/mlops_ws/LAW/projects/mmdet3d_plugin

# mmdetection3dをインストール
echo "mmdetection3dをインストールします..."
cd /home/shunya.seiya/mlops_ws
if [ ! -d "mmdetection3d" ]; then
    git clone https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    git checkout -f v0.17.1
    pip install -v -e .
else
    echo "mmdetection3dは既にクローンされています。"
    cd mmdetection3d
    git checkout -f v0.17.1
    pip install -v -e .
fi

echo "インストールしたパッケージの確認:"
pip list | grep mmseg
pip list | grep mmdet

echo "依存関係のインストールが完了しました。" 