#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import pickle
import mmcv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import cv2
import argparse
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
import os.path as osp

from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import init_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

# LAWモデルのプラグインをインポート
import sys
sys.path.append('/home/shunya.seiya/mlops_ws/LAW')
import projects.mmdet3d_plugin.LAW

# カラーマップ定義
DETECTION_COLORS = {
    'car': (0, 1, 0),          # 緑
    'truck': (0, 0.8, 0.2),    # 緑青
    'bus': (0, 0.6, 0.4),      # ターコイズ
    'trailer': (0, 0.4, 0.6),  # 青緑
    'construction_vehicle': (0, 0.2, 0.8), # 青紫
    'pedestrian': (1, 0.8, 0), # 黄色
    'motorcycle': (1, 0, 0),   # 赤
    'bicycle': (1, 0, 1),      # マゼンタ
    'traffic_cone': (0.8, 0.4, 0), # オレンジ
    'barrier': (0.6, 0.6, 0.6)  # グレー
}

# GT用カラー（少し薄め）
GT_COLORS = {k: (v[0]*0.7+0.3, v[1]*0.7+0.3, v[2]*0.7+0.3) for k, v in DETECTION_COLORS.items()}

# コマンドカラー
COMMAND_COLORS = {
    '左折': 'blue',
    '直進': 'green',
    '右折': 'red',
    '停止': 'orange'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize LAW predictions')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--data-root', default='data/nuscenes/', help='NuScenes data root')
    parser.add_argument('--scene-token', help='Specific scene token to visualize')
    parser.add_argument('--output-dir', default='./visualization_results', help='Output directory')
    parser.add_argument('--max-frames', type=int, default=50, help='Maximum number of frames to visualize')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    args = parser.parse_args()
    return args

from pyquaternion import Quaternion
import numpy as np

def get_lidar_gt(nusc, sample):
    """NuScenesのannotationをLiDAR座標系に変換した GT bboxes & labels を返す"""
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_token)

    # 1) ego pose (global->ego)
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ego_q = Quaternion(ego_pose['rotation'])
    ego_t = np.array(ego_pose['translation'])

    # 2) lidar calibration (ego->lidar)
    calib = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_q = Quaternion(calib['rotation'])
    lidar_t = np.array(calib['translation'])

    ann_tokens = sample['anns']
    bboxes_3d = []
    labels_3d = []

    for ann_token in ann_tokens:
        ann = nusc.get('sample_annotation', ann_token)
        # カテゴリ名からクラスIDを取得する (適宜自分のモデルのクラス定義に合わせる)
        cat_name = ann['category_name']
        label_id = convert_category_to_label(cat_name)
        if label_id < 0:
            # 学習対象でないカテゴリはスキップ
            continue

        # 3) box center / rotation (global)
        center_global = np.array(ann['translation'])
        q_global = Quaternion(ann['rotation'])

        # ---- global -> ego ----
        #center_ego = (ego_q.inverse * (center_global - ego_t))
        center_ego = ego_q.inverse.rotate(center_global - ego_t)
        q_ego = ego_q.inverse * q_global

        # ---- ego -> lidar ----
        #center_lidar = (lidar_q.inverse * (center_ego - lidar_t))
        center_lidar = lidar_q.inverse.rotate(center_ego - lidar_t)
        q_lidar = lidar_q.inverse * q_ego

        # サイズ(dx, dy, dz)
        dx, dy, dz = ann['size']

        # 回転行列→yaw
        rot_mat = q_lidar.rotation_matrix
        yaw_lidar = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])

        # 最終 [x, y, z, dx, dy, dz, yaw]
        box_3d = [center_lidar[0], center_lidar[1], center_lidar[2],
                  dx, dy, dz, yaw_lidar]
        bboxes_3d.append(box_3d)
        labels_3d.append(label_id)

    if len(bboxes_3d) == 0:
        bboxes_3d = np.zeros((0, 7), dtype=np.float32)
        labels_3d = np.zeros((0,), dtype=np.int64)
    else:
        bboxes_3d = np.array(bboxes_3d, dtype=np.float32)
        labels_3d = np.array(labels_3d, dtype=np.int64)

    return bboxes_3d, labels_3d

def convert_category_to_label(cat_name):
    """
    カテゴリ名をモデルのクラスIDに変換するサンプル。
    NuScenesには 'vehicle.car', 'vehicle.truck', など階層型カテゴリ名が多い。
    学習時のクラスリストに合わせて適宜対応付けを調整してください。
    """
    # 例として10クラス分をざっくり書く
    nuscenes_classes = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    for i, c in enumerate(nuscenes_classes):
        if c in cat_name:
            return i
    return -1  # 見つからなければ -1


def build_model_from_cfg(config_path, checkpoint_path, device='cuda:0'):
    """モデルを構築し、重みをロードする"""
    cfg = Config.fromfile(config_path)
    
    # データセット設定の調整
    cfg.model.pretrained = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        cfg.data.test.pop('samples_per_gpu', 1)
    
    # モデルの構築
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # FP16の設定
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    # チェックポイントのロード
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    # クラス情報の設定
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    
    # デバイスの設定
    model = MMDataParallel(model, device_ids=[0])
    model.to(device)
    model.eval()
    
    return model, cfg

def prepare_input_data(nusc, sample, cfg):
    """入力データの準備"""
    # サンプルデータの取得
    sample_data = {}
    
    # LiDARデータの取得
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_path = osp.join(nusc.dataroot, lidar_data['filename'])
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    
    # カメラデータの取得
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    imgs = []
    img_metas = []
    for cam_name in cam_names:
        cam_token = sample['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)
        img_path = osp.join(nusc.dataroot, cam_data['filename'])
        img = mmcv.imread(img_path)
        
        # カメラの内部パラメータと外部パラメータ
        cam_info = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsic = np.array(cam_info['camera_intrinsic'])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = Quaternion(cam_info['rotation']).rotation_matrix
        extrinsic[:3, 3] = np.array(cam_info['translation'])
        
        imgs.append(img)
        img_metas.append({
            'filename': img_path,
            'cam2img': intrinsic,
            'lidar2cam': extrinsic,
        })
    
    # エゴ車の情報
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    bboxes_3d, labels_3d = get_lidar_gt(nusc, sample)
    
    # 入力データの形式を整える
    input_dict = {
        'points': points,
        'img': imgs,
        'img_metas': [img_metas],
        'ego_pose': ego_pose,
        'gt_bboxes_3d': [bboxes_3d],   # リスト化しておく (mmdet3dの慣習)
        'gt_labels_3d': [labels_3d],
    }
    
    return input_dict

def inference_single_frame(model, input_dict, device='cuda:0'):
    """単一フレームの推論を実行"""
    with torch.no_grad():
        # データをGPUに転送
        if 'points' in input_dict:
            points = torch.from_numpy(input_dict['points']).float().to(device)
            input_dict['points'] = [points]
        
        if 'img' in input_dict:
            imgs = [torch.from_numpy(img).float().to(device) for img in input_dict['img']]
            input_dict['img'] = imgs
        # ここで gt_bboxes_3d, gt_labels_3d もGPUに送る
        if 'gt_bboxes_3d' in input_dict:
            gt_bboxes_3d_list = []
            for b in input_dict['gt_bboxes_3d']:
                gt_bboxes_3d_list.append(torch.from_numpy(b).float().to(device))
            input_dict['gt_bboxes_3d'] = gt_bboxes_3d_list
 
        if 'gt_labels_3d' in input_dict:
            gt_labels_3d_list = []
            for l in input_dict['gt_labels_3d']:
                gt_labels_3d_list.append(torch.from_numpy(l).long().to(device))
            input_dict['gt_labels_3d'] = gt_labels_3d_list
        # 推論実行
        result = model(return_loss=False, rescale=True, **input_dict)
        
    return result

def get_ego_command(img_meta):
    """運転コマンドを取得"""
    ego_fut_cmd = img_meta.get('ego_fut_cmd', None)
    if ego_fut_cmd is None:
        return "不明"
    
    if torch.is_tensor(ego_fut_cmd):
        ego_fut_cmd = ego_fut_cmd.cpu().numpy()
    
    if ego_fut_cmd.shape[-1] >= 4:
        cmd_idx = np.argmax(ego_fut_cmd[0, 0])
        commands = ['左折', '直進', '右折', '停止']
        return commands[cmd_idx]
    return "不明"

def draw_box_3d(ax, box, color=(1, 0, 0), label=None):
    """3Dボックスを描画"""
    corners = box.corners()
    for i in range(4):
        ax.plot([corners[0, i], corners[0, i+4]], 
                [corners[1, i], corners[1, i+4]], 
                [corners[2, i], corners[2, i+4]], color=color, alpha=0.5)
        ax.plot([corners[0, i], corners[0, (i+1)%4]], 
                [corners[1, i], corners[1, (i+1)%4]], 
                [corners[2, i], corners[2, (i+1)%4]], color=color, alpha=0.5)
        ax.plot([corners[0, i+4], corners[0, ((i+1)%4)+4]], 
                [corners[1, i+4], corners[1, ((i+1)%4)+4]], 
                [corners[2, i+4], corners[2, ((i+1)%4)+4]], color=color, alpha=0.5)
    
    # ラベルがあれば描画
    if label:
        center = box.center
        ax.text(center[0], center[1], center[2] + 1, label, color=color)

def draw_trajectory(ax, traj, color='blue', linestyle='-', label=None):
    """軌跡を描画"""
    ax.plot(traj[:, 0], traj[:, 1], linestyle, color=color, linewidth=2, label=label)
    
    # 最終位置に矢印
    if len(traj) > 1:
        direction = traj[-1] - traj[-2]
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            ax.arrow(traj[-1, 0], traj[-1, 1], direction[0], direction[1], 
                    head_width=0.5, head_length=0.7, fc=color, ec=color)

def project_lidar_to_img(lidar_points, cam_intrinsic, cam_extrinsic, img_shape):
    """LIDARポイントをカメラ画像に投影"""
    # 4xNの均質座標に変換
    points = np.vstack((lidar_points[:, :3].T, np.ones(lidar_points.shape[0])))
    
    # LiDARからカメラへの変換
    points = cam_extrinsic @ points
    
    # 画像平面への投影
    points = cam_intrinsic @ points[:3, :]
    points = points / points[2:3, :]  # 正規化
    
    # 画像内のポイントのフィルタリング
    mask = (points[0, :] >= 0) & (points[0, :] < img_shape[1]) & \
           (points[1, :] >= 0) & (points[1, :] < img_shape[0]) & \
           (points[2, :] > 0)
    
    return points[:2, mask].T, mask

def visualize_frame(nusc, sample_token, model, cfg, output_path):
    """1フレームの可視化を行う"""
    # サンプルデータ取得
    sample = nusc.get('sample', sample_token)
    
    # 入力データの準備
    input_dict = prepare_input_data(nusc, sample, cfg)
    
    # 推論実行
    result = inference_single_frame(model, input_dict)
    
    # キャリブレーションデータとイメージデータの取得
    cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    cam_front_img = Image.open(osp.join(nusc.dataroot, cam_front_data['filename']))
    img_shape = cam_front_img.size[::-1]  # (H, W)
    
    # CAM_FRONTのキャリブレーション
    cam_intrinsic = np.array(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])['camera_intrinsic'])
    cam_extrinsic = np.eye(4)
    cam_rotation = Quaternion(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])['rotation']).rotation_matrix
    cam_translation = np.array(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])['translation'])
    cam_extrinsic[:3, :3] = cam_rotation
    cam_extrinsic[:3, 3] = cam_translation
    
    # エゴポーズ
    ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
    ego_rotation = Quaternion(ego_pose['rotation']).rotation_matrix
    ego_translation = np.array(ego_pose['translation'])
    
    # LIDARデータ取得
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_path = osp.join(nusc.dataroot, lidar_data['filename'])
    lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    
    # LIDARからエゴ座標系への変換
    lidar_to_ego = np.eye(4)
    lidar_rotation = Quaternion(nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])['rotation']).rotation_matrix
    lidar_translation = np.array(nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])['translation'])
    lidar_to_ego[:3, :3] = lidar_rotation
    lidar_to_ego[:3, 3] = lidar_translation
    
    # GTアノテーションの取得
    annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
    
    # 自車両の軌跡データ（GT+予測）
    # 実際のデータに合わせて調整が必要
    ego_gt_trajectory = None
    ego_pred_trajectory = None
    if 'trajectories' in result:
        traj_data = result['trajectories'].get(sample_token, {})
        ego_gt_trajectory = traj_data.get('gt', None)
        ego_pred_trajectory = traj_data.get('pred', None)
    
    # 運転コマンド（LAWから取得）
    command = get_ego_command(sample['data']['CAM_FRONT'])
    
    # 可視化
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. カメラ画像
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(cam_front_img)
    ax_img.set_title('Camera Front')
    ax_img.axis('off')
    
    # 2. 上面図（BEV）- 点群とボックス
    ax_bev = fig.add_subplot(gs[1:, :2])
    # 点群プロット（密度が高いため間引く）
    downsample_factor = 10
    sparse_points = lidar_points[::downsample_factor, :2]  # x, y座標のみ使用
    ax_bev.scatter(sparse_points[:, 0], sparse_points[:, 1], s=0.5, c='lightgrey', alpha=0.5)
    
    # 3. 側面図
    ax_side = fig.add_subplot(gs[1, 2])
    ax_side.scatter(lidar_points[::downsample_factor, 0], lidar_points[::downsample_factor, 2], s=0.5, c='lightgrey', alpha=0.5)
    ax_side.set_title('Side View')
    ax_side.set_xlabel('X (m)')
    ax_side.set_ylabel('Z (m)')
    ax_side.set_xlim(-40, 40)
    ax_side.set_ylim(-5, 10)
    
    # 4. 後面図
    ax_rear = fig.add_subplot(gs[2, 2])
    ax_rear.scatter(lidar_points[::downsample_factor, 1], lidar_points[::downsample_factor, 2], s=0.5, c='lightgrey', alpha=0.5)
    ax_rear.set_title('Rear View')
    ax_rear.set_xlabel('Y (m)')
    ax_rear.set_ylabel('Z (m)')
    ax_rear.set_xlim(-40, 40)
    ax_rear.set_ylim(-5, 10)
    
    # GTボックスを描画
    for ann in annotations:
        category = ann['category_name']
        if category in DETECTION_COLORS:
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            color = GT_COLORS[category]
            # BEVに描画
            corners = box.bottom_corners()
            ax_bev.plot([corners[0, 0], corners[0, 1], corners[0, 2], corners[0, 3], corners[0, 0]],
                      [corners[1, 0], corners[1, 1], corners[1, 2], corners[1, 3], corners[1, 0]],
                      '-', color=color, alpha=0.5, linewidth=1)
            # 側面図に描画
            ax_side.plot([corners[0, 0], corners[0, 1], corners[0, 2], corners[0, 3], corners[0, 0]],
                       [corners[2, 0], corners[2, 1], corners[2, 2], corners[2, 3], corners[2, 0]],
                       '-', color=color, alpha=0.5, linewidth=1)
            # 後面図に描画
            ax_rear.plot([corners[1, 0], corners[1, 1], corners[1, 2], corners[1, 3], corners[1, 0]],
                       [corners[2, 0], corners[2, 1], corners[2, 2], corners[2, 3], corners[2, 0]],
                       '-', color=color, alpha=0.5, linewidth=1)
    
    # 予測ボックスを描画
    if result:
        for pred in result:
            category = pred['name']
            if category in DETECTION_COLORS:
                box = Box(pred['translation'], pred['size'], Quaternion(pred['rotation']))
                color = DETECTION_COLORS[category]
                # BEVに描画
                corners = box.bottom_corners()
                ax_bev.plot([corners[0, 0], corners[0, 1], corners[0, 2], corners[0, 3], corners[0, 0]],
                          [corners[1, 0], corners[1, 1], corners[1, 2], corners[1, 3], corners[1, 0]],
                          '-', color=color, linewidth=2)
                # 側面図に描画
                ax_side.plot([corners[0, 0], corners[0, 1], corners[0, 2], corners[0, 3], corners[0, 0]],
                           [corners[2, 0], corners[2, 1], corners[2, 2], corners[2, 3], corners[2, 0]],
                           '-', color=color, linewidth=2)
                # 後面図に描画
                ax_rear.plot([corners[1, 0], corners[1, 1], corners[1, 2], corners[1, 3], corners[1, 0]],
                           [corners[2, 0], corners[2, 1], corners[2, 2], corners[2, 3], corners[2, 0]],
                           '-', color=color, linewidth=2)
    
    # 軌跡を描画
    if ego_gt_trajectory is not None:
        draw_trajectory(ax_bev, ego_gt_trajectory, color='black', linestyle='-', label='GT Trajectory')
    
    if ego_pred_trajectory is not None:
        draw_trajectory(ax_bev, ego_pred_trajectory, color='blue', linestyle='--', label='Predicted Trajectory')
    
    # コマンド情報を表示
    ax_bev.text(0.02, 0.98, f"Command: {command}", transform=ax_bev.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 凡例を追加
    legend_patches = []
    for name, color in DETECTION_COLORS.items():
        legend_patches.append(mpatches.Patch(color=color, label=name))
    
    # GTとPredictionの凡例
    legend_patches.append(mpatches.Patch(color='black', label='GT Trajectory'))
    legend_patches.append(mpatches.Patch(color='blue', label='Pred Trajectory'))
    
    ax_bev.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # BEVの設定
    ax_bev.set_title('Bird\'s Eye View')
    ax_bev.set_xlabel('X (m)')
    ax_bev.set_ylabel('Y (m)')
    ax_bev.set_xlim(-40, 40)
    ax_bev.set_ylim(-40, 40)
    ax_bev.grid(True)
    # 自車位置を原点に
    ax_bev.plot(0, 0, 'ko', markersize=8)
    
    # タイムスタンプ情報
    timestamp = nusc.get('sample', sample_token)['timestamp'] / 1000000  # マイクロ秒からに秒変換
    fig.suptitle(f'Frame: {sample_token} | Time: {timestamp:.2f}s', fontsize=16)
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def main():
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'frames'), exist_ok=True)
    
    # NuScenesデータセットの初期化
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.data_root, verbose=True)
    
    # モデルの構築
    model, cfg = build_model_from_cfg(args.config, args.checkpoint, device=args.device)
    
    # シーンの選択
    if args.scene_token:
        scene = nusc.get('scene', args.scene_token)
    else:
        scene = np.random.choice(nusc.scene)
    
    print(f"Processing scene: {scene['name']}")
    
    # シーン内の各フレームを処理
    sample_token = scene['first_sample_token']
    frame_count = 0
    
    while sample_token and frame_count < args.max_frames:
        print(f"Processing frame {frame_count + 1}/{args.max_frames}")
        
        # フレームの出力パス
        frame_path = osp.join(args.output_dir, 'frames', f'frame_{frame_count:06d}.png')
        
        # フレームの可視化
        visualize_frame(nusc, sample_token, model, cfg, frame_path)
        
        # 次のフレームへ
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']
        frame_count += 1
    
    # 動画の生成
    print("Generating video...")
    frame_pattern = osp.join(args.output_dir, 'frames', 'frame_%06d.png')
    video_path = osp.join(args.output_dir, f'visualization_{scene["token"]}.mp4')
    
    os.system(f'ffmpeg -y -framerate 10 -i {frame_pattern} -c:v libx264 -pix_fmt yuv420p {video_path}')
    
    print(f"Visualization complete! Video saved to: {video_path}")

if __name__ == '__main__':
    main() 