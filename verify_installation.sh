#!/bin/bash

# Conda環境を有効化
echo "Conda環境を有効化します..."
source /home/shunya.seiya/miniconda3/etc/profile.d/conda.sh
conda activate law
echo "Conda環境の有効化完了: $(conda info --envs | grep '*')"

# モジュールインポートテスト
echo "モジュールインポートテスト:"

# インポートテスト関数
test_import() {
    local module=$1
    echo -n "  $module: "
    python -c "
try:
    import $module
    print('OK, version:', $module.__version__)
except ImportError as e:
    print('FAILED:', str(e))
except AttributeError:
    print('OK, but no version attribute')
"
}

# 各モジュールのテスト
test_import "torch"
test_import "mmcv"
test_import "mmdet"
test_import "mmseg"
test_import "mmdet3d"

# PYTHONPATHの確認
echo -e "\nPYTHONPATH確認:"
echo "$PYTHONPATH"

# パス情報の確認
echo -e "\nPython sys.path確認:"
python -c "
import sys
for i, path in enumerate(sys.path):
    print(f'  {i+1}. {path}')
"

# インストールされたパッケージの一覧
echo -e "\nインストールされたパッケージ一覧:"
pip list | grep -E "torch|mmcv|mmdet|mmseg|nuscenes|timm|yapf"

# mmdetection3dのディレクトリパス確認
echo -e "\nMMDetection3Dディレクトリ確認:"
find /home/shunya.seiya/mlops_ws -name "mmdet3d" -type d 