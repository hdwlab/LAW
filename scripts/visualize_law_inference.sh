#!/bin/bash

# LAWモデルの推論結果を可視化するスクリプト

# SLURM設定
#SBATCH --job-name=LAW_vis_inf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH -e law_vis_inf_%j.err
#SBATCH -o law_vis_inf_%j.out

# デバッグ情報の有効化
export PYTHONWARNINGS=default
export CUDA_LAUNCH_BLOCKING=1

# 環境変数
export PYTHONPATH=/home/shunya.seiya/mlops_ws/LAW:$PYTHONPATH
export PYTHONPATH=/home/shunya.seiya/mlops_ws/mmdetection3d:$PYTHONPATH
export PYTHONPATH=/home/shunya.seiya/mlops_ws/LAW/projects/mmdet3d_plugin:$PYTHONPATH

# CUDA設定
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Conda環境の有効化
source /home/shunya.seiya/miniconda3/bin/activate law

# 作業ディレクトリに移動
cd /home/shunya.seiya/mlops_ws/LAW

# 環境情報の出力
echo "=== Environment Information ==="
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA version:"
python -c "import torch; print(torch.version.cuda)"
echo "GPU information:"
nvidia-smi
echo "==========================="

# 設定ファイルとチェックポイント
CONFIG=projects/configs/law/default.py
CHECKPOINT=checkpoints/perception-free/LAW.pth
DATA_ROOT=data/nuscenes/

# 出力ディレクトリの設定
OUTPUT_DIR=visualization_inference_results
mkdir -p $OUTPUT_DIR

# 推論結果ファイルのパス
RESULT_PATH=results/law_inference_results.pkl

# 可視化の実行
echo "Starting visualization of LAW inference results..."

python tools/visualize_law_inference.py \
    --result-path $RESULT_PATH \
    --data-root $DATA_ROOT \
    --version v1.0-trainval \
    --save-path $OUTPUT_DIR \
    --max-samples 50 \
    --framerate 10

echo "Visualization process completed!"
echo "Results saved to: $OUTPUT_DIR/law_visualization.mp4"
