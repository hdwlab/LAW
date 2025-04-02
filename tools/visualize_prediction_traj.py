#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import argparse
import numpy as np
import mmcv
import torch
import matplotlib.pyplot as plt

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model

# LAWのimport (事前に修正済みのLAW.pyがある想定)
import sys
sys.path.append('/home/shunya.seiya/mlops_ws/LAW')
import projects.mmdet3d_plugin.LAW

def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for LAW inference using NuScenes dataloader'
    )
    # config + checkpoint
    parser.add_argument('config', help='config file for LAW (NuScenes) test')
    parser.add_argument('checkpoint', help='checkpoint file')

    # ユーザー指定の追加引数
    parser.add_argument('--data-root', default='data/nuscenes', help='NuScenes root path')
    parser.add_argument('--version', default='v1.0-trainval', help='NuScenes version string')
    parser.add_argument('--scene-token', default='', help='Scene token (currently not used in this script)')
    parser.add_argument('--max-frames', type=int, default=50, help='Process up to N samples')
    parser.add_argument('--output-dir', default='demo_out', help='directory to save frames & video')
    parser.add_argument('--framerate', type=int, default=10, help='fps for generated video')

    # device
    parser.add_argument('--device', default='cuda:0', help='CUDA device or CPU')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 1) Config読み込み ==========
    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None

    # データルートなどを上書きしたい場合は、configの構造に合わせて書き換え
    # 例:
    # if 'dataset' in cfg.data.test:
    #     cfg.data.test.dataset.data_root = args.data_root
    # など
    # 今回は最小限のため省略

    # テストモードにする & バッチサイズ1に固定
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        cfg.data.test.pop('samples_per_gpu', None)

    # ========== 2) Dataset構築 ==========
    dataset = build_dataset(cfg.data.test)

    # ========== 3) DataLoader構築 ==========
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,  # batch size 1
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # ========== 4) モデル構築 ==========
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    model = MMDataParallel(model, device_ids=[0])
    model.to(args.device)
    model.eval()

    # 出力用ディレクトリ
    frames_dir = osp.join(args.output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # ========== 5) データをループして推論 ==========
    frame_count = 0
    for i, data in enumerate(data_loader):
        if frame_count >= args.max_frames:
            break

        print(f"[INFO] Processing sample {frame_count}")

        with torch.no_grad():
            # forward_test() を呼ぶ
            result = model(return_loss=False, rescale=True, **data)

        # result: list[dict], バッチサイズ1 => result[0]
        single_result = result[0]

        # 軌跡の取得
        gt_xy = None
        pred_xy = None
        if 'trajectories' in single_result:
            trajs = single_result['trajectories']
            gt3d = trajs.get('gt', None)[0]
            pred3d = trajs.get('pred', None)
            print("pred3d is not None. shape =", pred3d.shape, "ndim=", pred3d.ndim)
            print(">> GT shape:", None if gt3d is None else gt3d.shape)
            # もし shape=(T,3) なら 2D軌跡として x,y だけ使う
            if gt3d is not None and gt3d.shape[1] >= 2:
                gt_xy = gt3d[:, :2]
            if pred3d is not None and pred3d.shape[1] >= 2:
                pred_xy = pred3d[:, :2]

        # 可視化: 軌跡を2Dプロット
        out_path = osp.join(frames_dir, f"frame_{frame_count:06d}.png")
        plt.figure(figsize=(6,6))
        plt.title(f"Frame {frame_count} - LAW Trajectories")
        if gt_xy is not None and len(gt_xy) > 1:
            plt.plot(gt_xy[:,0], gt_xy[:,1], 'g-o', label='GT')
        if pred_xy is not None and len(pred_xy) > 1:
            plt.plot(pred_xy[:,0], pred_xy[:,1], 'r--o', label='Pred')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        plt.legend()
        plt.grid(True)

        plt.savefig(out_path)
        plt.close()

        frame_count += 1

    # ========== 6) ffmpegで動画作成 ==========
    print(f"[INFO] Generating video with framerate={args.framerate} ...")
    frame_pattern = osp.join(frames_dir, "frame_%06d.png")
    video_path = osp.join(args.output_dir, "result_video.mp4")

    os.system(f"ffmpeg -y -framerate {args.framerate} -i {frame_pattern} -c:v libx264 -pix_fmt yuv420p {video_path}")
    print("[INFO] Done! Video saved to:", video_path)

if __name__ == '__main__':
    main()
