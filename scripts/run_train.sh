#!/bin/bash
#SBATCH --job-name=LAW_test
#SBATCH --output=LAW_test_%j.out
#SBATCH --error=LAW_test_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu01
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4

# 作業ディレクトリに移動
cd /home/shunya.seiya/mlops_ws/LAW

# データディレクトリのシンボリックリンク作成
mkdir -p data/nuscenes
ln -sf /home/shunya.seiya/data/nuscenes/law_nuscenes_infos_temporal_train.pkl data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/law_nuscenes_infos_temporal_val.pkl data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/samples data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/sweeps data/nuscenes/
ln -sf /home/shunya.seiya/data/nuscenes/v1.0-trainval data/nuscenes/

# mapsディレクトリへのリンク追加
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
#sqexport CUDA_VISIBLE_DEVICES=2,3

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
echo "Starting evaluation of LAW model..."
# LAW-Base モデルでの評価を実行（必要に応じてパスを変更）
#python tools/test.py projects/configs/LAW/LAW_base_stage_2.py checkpoints/LAW_base.pth --launcher none --eval bbox --tmpdir tmp

# 可視化の実行（オプション）
echo "Generating visualization..."
#python tools/analysis_tools/visualization.py --result-path /home/shunya.seiya/mlops_ws/LAW/test/LAW_base_stage_2/Fri_Mar_21_09_13_45_2025/pts_bbox/results_nusc.pkl --save-path results

python -m torch.distributed.run --nproc_per_node=2 --master_port=2345 tools/train.py G --launcher pytorch --deterministic --work-dir outputs
echo "Testing completed!"
