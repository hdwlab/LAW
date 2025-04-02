#!/bin/bash
#SBATCH --job-name=LAW_inference
#SBATCH --output=LAW_inference_%j.out
#SBATCH --error=LAW_inference_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu01
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

# 作業ディレクトリに移動
cd /home/shunya.seiya/mlops_ws/LAW

# データディレクトリの設定（相対パスで）
mkdir -p data/nuscenes
ln -sf /home/shunya.seiya/data/nuscenes/vad_nuscenes_infos_temporal_train.pkl data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/vad_nuscenes_infos_temporal_val.pkl data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/samples data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/sweeps data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/v1.0-trainval data/nuscenes/

# mapsディレクトリへのリンクも追加（必要な場合）
mkdir -p data/nuscenes/maps
ln -sf /mnt/hdl_work1/share/database/nuscenes/maps/* data/nuscenes/maps/

# CANバスデータ用ディレクトリ
rm -rf data/can_bus
ln -sf /home/shunya.seiya/data/can_bus data/

# Conda環境を有効化
echo "Conda環境を有効化します..."
source /home/shunya.seiya/miniconda3/etc/profile.d/conda.sh
conda activate law
echo "Conda環境の有効化完了"

# CUDA 11.1を使用するように環境変数を設定
echo "CUDA 11.1を設定します..."
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# CUDAデバイスとバージョンを確認
echo "CUDA設定の確認:"
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    $CUDA_HOME/bin/nvcc --version
else
    echo "警告: nvccが見つかりません"
fi
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# PYTHONPATHにmmdet3dディレクトリを追加
echo "PYTHONPATHの設定:"
export PYTHONPATH=$PYTHONPATH:/home/shunya.seiya/mlops_ws/mmdetection3d
export PYTHONPATH=$PYTHONPATH:/home/shunya.seiya/mlops_ws/LAW/projects/mmdet3d_plugin
echo "PYTHONPATH: $PYTHONPATH"

# モジュールインポートテスト
echo "モジュールインポートテスト:"
python -c "
import sys
print('Python path:', sys.path)
print('\\nImport test:')
try:
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA version:', torch.version.cuda)
        print('GPU device:', torch.cuda.get_device_name(0))
except ImportError as e:
    print('Failed to import torch:', e)

try:
    import mmdet
    print('mmdet imported successfully, version:', mmdet.__version__)
except ImportError as e:
    print('Failed to import mmdet:', e)

try:
    import mmcv
    print('mmcv imported successfully, version:', mmcv.__version__)
except ImportError as e:
    print('Failed to import mmcv:', e)

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
    print('mmdet3dの検索パス:')
    for path in sys.path:
        if 'mmdet' in path:
            print('  -', path)
"

# GPUの使用状況を表示
echo "Using GPU:"
nvidia-smi

# パッケージのバージョンを確認
echo "インストール済みパッケージの確認:"
pip list | grep torch
pip list | grep mmcv
pip list | grep mmdet
pip list | grep mmseg
pip list | grep nuscenes
pip list | grep timm
pip list | grep yapf

# mmdetection3dのパスを確認
echo "mmdetection3dのパス確認:"
find /home/shunya.seiya/mlops_ws -name "mmdet3d" -type d

# 結果保存用ディレクトリの作成
mkdir -p results

# 推論の実行
echo "Starting inference with LAW model (CUDA 11.1)..."
# GPUモードで実行（結果を保存・評価するオプションを追加）
python tools/test.py projects/configs/law/default.py checkpoints/perception-free/LAW.pth --launcher none --out results/result.pkl --eval bbox

echo "Inference completed!" 