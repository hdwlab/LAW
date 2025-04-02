#!/bin/bash

# Conda環境を有効化
echo "Conda環境を有効化します..."
source /home/shunya.seiya/miniconda3/etc/profile.d/conda.sh
conda activate law
echo "Conda環境の有効化完了: $(conda info --envs | grep '*')"

# mmdetection3dディレクトリに移動
cd /home/shunya.seiya/mlops_ws/mmdetection3d

# 現在のディレクトリを確認
echo "現在のディレクトリ: $(pwd)"

# mmdet3dをPYTHONPATHに追加
echo "mmdetection3dをPYTHONPATHに追加します..."
export PYTHONPATH=$PYTHONPATH:/home/shunya.seiya/mlops_ws/mmdetection3d

# pip installで明示的にインストール
echo "pip installでmmdetection3dを再インストールします..."
pip install -e .

# インストール確認
echo "インポートテスト:"
python -c "
try:
    import mmdet3d
    print('mmdet3d インポート成功! バージョン:', mmdet3d.__version__)
except ImportError as e:
    print('mmdet3d インポート失敗:', e)
    print('PYTHONPATHを確認します...')
    import sys
    print('Python path:', sys.path)
"

echo "修正処理が完了しました。" 