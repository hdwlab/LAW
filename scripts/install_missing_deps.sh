#!/bin/bash

# Conda環境を有効化
echo "Conda環境を有効化します..."
source /home/shunya.seiya/miniconda3/etc/profile.d/conda.sh
conda activate law
echo "Conda環境の有効化完了: $(conda info --envs | grep '*')"

# 4. Install MMDetection and MMSegmentation
echo "MMSegmentationをインストールします..."
pip install mmsegmentation==0.14.1

# 5. Install MMDetection3D
echo "MMDetection3Dをインストールします..."
cd /home/shunya.seiya/mlops_ws
if [ ! -d "mmdetection3d" ]; then
    echo "MMDetection3Dリポジトリをクローンします..."
    git clone https://github.com/open-mmlab/mmdetection3d.git
fi

cd mmdetection3d
echo "MMDetection3D v0.17.1にチェックアウトします..."
git checkout -f v0.17.1
echo "MMDetection3Dをインストールします..."
python setup.py develop

# インストール確認
echo "インストールされたパッケージを確認します..."
pip list | grep mmseg
pip list | grep mmdet

# Pythonでインポートテスト
echo "モジュールインポートテスト:"
python -c "
try:
    import mmseg
    print('mmseg imported successfully, version:', mmseg.__version__)
except ImportError as e:
    print('Failed to import mmseg:', e)

try:
    import mmdet3d
    print('mmdet3d imported successfully, version:', mmdet3d.__version__)
except ImportError as e:
    print('Failed to import mmdet3d:', e)
"

echo "インストール処理が完了しました。エラーがなければ、すべてのパッケージが正常にインストールされています。" 