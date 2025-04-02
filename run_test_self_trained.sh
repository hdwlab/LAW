#!/bin/bash
#SBATCH --job-name=LAW_test_trained
#SBATCH --output=LAW_test_trained_%j.out
#SBATCH --error=LAW_test_trained_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu01
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# 作業ディレクトリに移動
cd /home/shunya.seiya/mlops_ws/LAW

# データディレクトリの設定
mkdir -p data/nuscenes
ln -sf /home/shunya.seiya/data/nuscenes/vad_nuscenes_infos_temporal_train.pkl data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/vad_nuscenes_infos_temporal_val.pkl data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/samples data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/sweeps data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/v1.0-trainval data/nuscenes/

# mapsディレクトリへのリンク
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

# CUDAデバイスの設定
export CUDA_VISIBLE_DEVICES=0

# PYTHONPATHの設定
echo "PYTHONPATHの設定:"
export PYTHONPATH=$PYTHONPATH:/home/shunya.seiya/mlops_ws/mmdetection3d
echo "PYTHONPATH: $PYTHONPATH"

# モジュールインポートテスト
echo "モジュールインポートテスト:"
python -c "
import sys
print('Python path:', sys.path)
print('\\nImport test:')
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

# テストの実行
echo "Starting test with LAW model..."
python tools/test_for_vis.py projects/configs/law/default.py outputs/latest.pth --launcher none --out results/result_self_trained.pkl

# 可視化の実行
#echo "Generating visualization..."
#python /home/shunya.seiya/mlops_ws/VAD/tools/analysis_tools/visualization.py --result-path results/result_self_trained.pkl --save-path visualization_results

echo "Testing completed!"
