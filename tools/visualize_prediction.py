#!/usr/bin/env python
# coding: utf-8

import math
import os
import os.path as osp
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mmcv

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model


# LAWをimport (修正済み)
import sys
sys.path.append('/home/shunya.seiya/mlops_ws/LAW')
import projects.mmdet3d_plugin.LAW

def parse_args():
    parser = argparse.ArgumentParser(description='Demo: 6 camera images + trajectory below')

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


def transform_xy(points_xy, car_x, car_y, car_yaw):
    """
    points_xy: shape=(N,2) の配列 (GTやPredの軌跡を想定)
    car_x, car_y: 自車両のグローバル座標
    car_yaw: 自車両の向き(ヨー角,ラジアン)
      (グローバル座標系での回転量; 前方が +X になるよう -car_yaw で回す)

    return: shape=(N,2) 自車両中心・前方+X軸のローカル座標に変換した結果
            points_xy が None の場合は None
    """
    if points_xy is None or len(points_xy) == 0:
        return None

    # 1) 平行移動 (車位置を原点に)
    shifted_x = points_xy[:, 0] - car_x
    shifted_y = points_xy[:, 1] - car_y

    # 2) 回転 (自車両が θ 回転しているなら -θ で逆回し)
    cos_t = math.cos(-car_yaw)
    sin_t = math.sin(-car_yaw)
    rot_x = shifted_x * cos_t - shifted_y * sin_t
    rot_y = shifted_x * sin_t + shifted_y * cos_t

    return np.column_stack((rot_x, rot_y))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Config
    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        cfg.data.test.pop('samples_per_gpu', None)

    # 2) Dataset & Dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # 3) Model
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

    # 出力先
    frames_dir = osp.join(args.output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    for i, data in enumerate(data_loader):
        if frame_count >= args.max_frames:
            break

        print(f"[INFO] sample {frame_count} ...")

        # 推論
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        single_result = result[0]

        # 軌跡
        gt_xy = None
        pred_xy = None
        if 'trajectories' in single_result:
            trajs = single_result['trajectories']
            gt3d = trajs.get('gt', None)[0]
            pred3d = trajs.get('pred', None)
            # 次元数チェック
            if gt3d is not None and gt3d.ndim >= 2 and gt3d.shape[1] >= 2:
                gt_xy = gt3d[:, :2]
            if pred3d is not None and pred3d.ndim >= 2 and pred3d.shape[1] >= 2:
                pred_xy = pred3d[:, :2]


        car_x, car_y, car_yaw = 0.0, 0.0, 0.0  # フォールバック値

        meta_frame0 = data['img_metas'].data[0][0][0]

        # 1) もし ego_fut_trajs[0][0] が 「現在の ego位置」 なら
        ego_fut_trajs = meta_frame0['ego_fut_trajs']  # shape=[1, future_steps, 2] かも
        car_x, car_y = ego_fut_trajs[0][0].tolist()  # 例: tensor([-0.1383, 4.7741]) -> ( -0.1383, 4.7741 )

        # 2) can_bus から yaw を取り出す (どのindexか要チェック)
        can_bus = meta_frame0['can_bus']
        yaw = can_bus[3]  # 仮に index=3 が yaw だと仮定

        # 3) gt_xy / pred_xy を transform
        gt_xy_local   = transform_xy(gt_xy, car_x, car_y, yaw)
        pred_xy_local = transform_xy(pred_xy, car_x, car_y, yaw)

        print("ego_pose=", car_x, car_y, car_yaw)

        # GT/予測軌跡を 自車両中心(0,0), 前方+Xにする
        gt_xy_local = transform_xy(gt_xy, car_x, car_y, car_yaw)
        pred_xy_local = transform_xy(pred_xy, car_x, car_y, car_yaw)


        # Figure: 3行x3列
        fig = plt.figure(figsize=(16,12))
        gs = GridSpec(3, 3, figure=fig)

        # A) 上2行に6カメラ画像
        # data['img']: list of Tensor => shape (B, num_cams, C, H, W) (batch=1)
        #imgs_container = data['img']            # DataContainer
        #imgs_tensor = imgs_container.data[0]    # shape=(B, num_cams, C, H, W)
        #imgs = imgs_tensor[0][0]                   # B=1 => shape=(num_cams, C, H, W)

        img_metas_container = data['img_metas']
        img_metas_list = img_metas_container.data[0]  
        # バッチサイズB=1を仮定 -> img_metas_list[0] が [dict_for_cam0, dict_for_cam1, ...]
        img_metas = img_metas_list[0]







        #num_cams = imgs.shape[0]  # typically 6
        #print("img_metas =", img_metas)
        #print("num_cams =", num_cams)
        meta_frame0 = img_metas[0]
        filenames_6cams = meta_frame0['filename'] 
        new_order_indices = [2, 0, 1, 5, 3, 4]  # 必要に応じて変更
        reordered_filenames = [filenames_6cams[i] for i in new_order_indices]

        # 今後は filenames_6cams の代わりに reordered_filenames を使う
        filenames_6cams = reordered_filenames

        num_cams = len(filenames_6cams)

        for idx in range(num_cams):
            # row=idx//3, col=idx%3
            row = idx // 3
            col = idx % 3
            ax_img = fig.add_subplot(gs[row, col])


            raw_bgr = mmcv.imread(filenames_6cams[idx])
            raw_rgb = mmcv.bgr2rgb(raw_bgr)
            ax_img.imshow(raw_rgb)
            ax_img.set_title(filenames_6cams[idx].split('/')[4], fontsize=8)
            ax_img.axis('off')

            """
            # Tensor -> NumPy
            img_tensor = imgs[idx].cpu().numpy()  # (3,H,W)
            img_norm_cfg = img_metas[idx]['img_norm_cfg']
            mean = np.array(img_norm_cfg['mean'], dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(img_norm_cfg['std'], dtype=np.float32).reshape(-1, 1, 1)
            to_rgb = img_norm_cfg.get('to_rgb', False)

            # 正規化を元に戻す
            img_tensor = img_tensor * std + mean
            # 値域を [0, 255] にクリップ
            img_tensor = np.clip(img_tensor, 0, 255)

            img_np = img_tensor.transpose(1,2,0).astype(np.uint8)  # (H,W,3)
            
            print("img_np shape=", img_np)
            # もしピクセルが0〜255でなければ適宜スケーリング

            #img_np = img_np[..., ::-1]
            # 表示
            ax_img.imshow(img_np)
            title_str = filenames_6cams[idx]  # or 'CAM_FRONT' etc.
            ax_img.set_title(title_str, fontsize=10)
            ax_img.axis('off')
            """

        # B) 最下行に軌跡を描画
        ax_traj = fig.add_subplot(gs[2, :])  # 3列分を1つに割り当て
        ax_traj.set_title("Trajectory (GT & Pred)")
        ax_traj.set_aspect('equal')
        ax_traj.set_xlim(-10, 10)
        ax_traj.set_ylim(-10, 10)
        ax_traj.set_xlabel("X")
        ax_traj.set_ylabel("Y")
        ax_traj.grid(True)

        #if gt_xy is not None and len(gt_xy) > 1:
        #    ax_traj.plot(gt_xy[:,0], gt_xy[:,1], 'g-o', label='GT')

        if gt_xy_local is not None and len(gt_xy_local) > 1:
            ax_traj.plot(gt_xy_local[:, 0], gt_xy_local[:, 1], 'g-o', label='GT')
            

        if pred_xy is not None and len(pred_xy) > 1:
            #ax_traj.plot(pred_xy[:,0], pred_xy[:,1], 'r--o', label='Pred')
            ax_traj.plot(pred_xy_local[:, 0], pred_xy_local[:, 1], 'r--o', label='Pred')


        ax_traj.legend()

        out_path = osp.join(frames_dir, f"frame_{frame_count:06d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close(fig)

        frame_count += 1

    # ffmpeg
    print(f"[INFO] Creating video with {args.framerate} fps ...")
    frame_pattern = osp.join(frames_dir, "frame_%06d.png")
    video_path = osp.join(args.output_dir, "merged_video.mp4")
    os.system(f"ffmpeg -y -framerate {args.framerate} -i {frame_pattern} -c:v libx264 -pix_fmt yuv420p {video_path}")
    print("[INFO] Done:", video_path)

if __name__ == '__main__':
    main()
