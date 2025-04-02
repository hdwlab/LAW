#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('')
import os
import argparse
import os.path as osp
from PIL import Image
from tqdm import tqdm
from typing import List, Dict

import cv2
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from mmdet.datasets.pipelines import to_tensor
from matplotlib.collections import LineCollection
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

# Import LAW specific modules
import sys
sys.path.append('/home/shunya.seiya/mlops_ws/LAW')
import projects.mmdet3d_plugin.LAW

# Define camera names
cams = ['CAM_FRONT',
 'CAM_FRONT_RIGHT',
 'CAM_BACK_RIGHT',
 'CAM_BACK',
 'CAM_BACK_LEFT',
 'CAM_FRONT_LEFT']

def color_map(values, cmap):
    """
    Convert values to colors using a colormap.
    """
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    cmap = plt.get_cmap(cmap)
    return cmap(norm(values))

def get_color(category_name: str):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    
    # Define a color map for different categories
    color_map = {
        'vehicle.bicycle': [255, 61, 99],
        'vehicle.bus.bendy': [255, 128, 0],
        'vehicle.bus.rigid': [255, 128, 0],
        'vehicle.car': [255, 158, 0],
        'vehicle.construction': [233, 150, 70],
        'vehicle.motorcycle': [255, 61, 99],
        'vehicle.trailer': [255, 158, 0],
        'vehicle.truck': [255, 128, 0],
        'human.pedestrian.adult': [0, 61, 255],
        'human.pedestrian.child': [0, 61, 255],
        'human.pedestrian.construction_worker': [0, 61, 255],
        'human.pedestrian.police_officer': [0, 61, 255],
        'movable_object.barrier': [70, 240, 240],
        'movable_object.trafficcone': [255, 61, 99],
    }
    
    if category_name == 'bicycle':
        return color_map['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return color_map['vehicle.construction']
    elif category_name == 'traffic_cone':
        return color_map['movable_object.trafficcone']

    for key in color_map.keys():
        if category_name in key:
            return color_map[key]
    return [0, 0, 0]

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matrix from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sensor2lidar_rotation = R.T  # points @ R.T + T
    sensor2lidar_translation = T

    return sensor2lidar_rotation, sensor2lidar_translation

def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
                       ):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Use the provided predicted annotations
    boxes = pred_anns
    
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic

def transform_xy(points_xy, car_x, car_y, car_yaw):
    """
    Transform points from global to local coordinates.
    
    Args:
        points_xy: shape=(N,2) array (GT or Pred trajectories)
        car_x, car_y: ego vehicle global coordinates
        car_yaw: ego vehicle orientation (yaw angle in radians)
            (global coordinate system rotation; rotate by -car_yaw to make front +X)
            
    Returns: 
        shape=(N,2) array of points transformed to local coordinates centered at the ego vehicle
        with front as +X axis. Returns None if points_xy is None or empty.
    """
    if points_xy is None or len(points_xy) == 0:
        return None

    # 1) Translate (move car position to origin)
    shifted_x = points_xy[:, 0] - car_x
    shifted_y = points_xy[:, 1] - car_y

    # 2) Rotate (if car is rotated by θ, rotate by -θ to align)
    cos_t = np.cos(-car_yaw)
    sin_t = np.sin(-car_yaw)
    rot_x = shifted_x * cos_t - shifted_y * sin_t
    rot_y = shifted_x * sin_t + shifted_y * cos_t

    return np.column_stack((rot_x, rot_y))

def visualize_law_predictions(sample_token, pred_data, out_path=None):
    """
    Visualize LAW predictions for a given sample.
    
    Args:
        sample_token: NuScenes sample token
        pred_data: Prediction data dictionary
        out_path: Output directory path
    """
    sample_rec = nusc.get('sample', sample_token)
    
    # Create figure with 3x3 grid (6 cameras + 1 BEV view)
    fig = plt.figure(figsize=(16, 12))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Visualize camera images (6 cameras)
    for idx, cam in enumerate(cams):
        row = idx // 3
        col = idx % 3
        ax_img = fig.add_subplot(gs[row, col])
        
        # Get camera data
        cam_token = sample_rec['data'][cam]
        cam_path = nusc.get_sample_data_path(cam_token)
        
        # Load and display image
        img = Image.open(cam_path)
        ax_img.imshow(img)
        ax_img.set_title(cam)
        ax_img.axis('off')
        
        # If we have trajectory predictions, overlay them on the front camera
        if cam == 'CAM_FRONT' and 'trajectories' in pred_data:
            # Get camera calibration
            sd_record = nusc.get('sample_data', cam_token)
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            
            # Get lidar to camera transformation
            lidar_token = sample_rec['data']['LIDAR_TOP']
            lidar_sd_record = nusc.get('sample_data', lidar_token)
            lidar_cs_record = nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
            lidar_pose_record = nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
            
            l2e_r = lidar_cs_record['rotation']
            l2e_t = lidar_cs_record['translation']
            e2g_r = lidar_pose_record['rotation']
            e2g_t = lidar_pose_record['translation']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix
            
            # Get sensor to lidar transformation
            s2l_r, s2l_t = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, 'camera')
            
            # Get lidar to image transformation
            lidar2cam_r = np.linalg.inv(s2l_r)
            lidar2cam_t = s2l_t @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            viewpad = np.eye(4)
            viewpad[:cam_intrinsic.shape[0], :cam_intrinsic.shape[1]] = cam_intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            
            # Project trajectories to image
            if 'pred' in pred_data['trajectories']:
                traj = pred_data['trajectories']['pred']
                
                # Add z-coordinate and homogeneous coordinate
                traj_3d = np.zeros((traj.shape[0], 4))
                traj_3d[:, 0:2] = traj[:, 0:2]  # x, y
                traj_3d[:, 2] = -1.0  # z (ground level)
                traj_3d[:, 3] = 1.0   # homogeneous coordinate
                
                # Project to image
                traj_img = lidar2img_rt @ traj_3d.T
                traj_img = traj_img[0:2, ...] / np.maximum(traj_img[2:3, ...], np.ones_like(traj_img[2:3, ...]) * 1e-5)
                traj_img = traj_img.T
                
                # Plot trajectory on image
                valid_idx = (traj_img[:, 0] >= 0) & (traj_img[:, 0] < img.width) & (traj_img[:, 1] >= 0) & (traj_img[:, 1] < img.height)
                if np.any(valid_idx):
                    ax_img.plot(traj_img[valid_idx, 0], traj_img[valid_idx, 1], 'r-', linewidth=2)
                    ax_img.scatter(traj_img[valid_idx, 0], traj_img[valid_idx, 1], c='r', s=15)
    
    # 2. Visualize BEV with trajectories
    ax_bev = fig.add_subplot(gs[2, :])
    ax_bev.set_title("Bird's Eye View with Trajectories")
    ax_bev.set_aspect('equal')
    ax_bev.set_xlim(-15, 15)
    ax_bev.set_ylim(-30, 30)
    ax_bev.grid(True)
    
    # Plot ego vehicle
    rect = plt.Rectangle((-1.5, -3.0), 3.0, 6.0, fill=True, color='gray', alpha=0.7)
    ax_bev.add_patch(rect)
    
    # Plot trajectories if available
    if 'trajectories' in pred_data:
        # Ground truth trajectory
        if 'gt' in pred_data['trajectories']:
            gt_traj = pred_data['trajectories']['gt']
            ax_bev.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-o', linewidth=2, markersize=4, label='Ground Truth')
        
        # Predicted trajectory
        if 'pred' in pred_data['trajectories']:
            pred_traj = pred_data['trajectories']['pred']
            ax_bev.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--o', linewidth=2, markersize=4, label='Prediction')
    
    # Add legend
    ax_bev.legend(loc='upper right')
    
    # Save figure
    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"{sample_token}_visualization.png"), dpi=200)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

def create_video_from_frames(frames_dir, output_path, framerate=10):
    """
    Create a video from a directory of frames using ffmpeg.
    
    Args:
        frames_dir: Directory containing the frames
        output_path: Path to save the output video
        framerate: Frames per second for the output video
    """
    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
    os.system(f"ffmpeg -y -framerate {framerate} -i {frame_pattern} -c:v libx264 -pix_fmt yuv420p {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize LAW inference results')
    parser.add_argument('--result-path', help='Path to LAW inference results file')
    parser.add_argument('--data-root', default='data/nuscenes', help='NuScenes data root')
    parser.add_argument('--version', default='v1.0-trainval', help='NuScenes dataset version')
    parser.add_argument('--save-path', help='Directory to save visualization results')
    parser.add_argument('--max-samples', type=int, default=50, help='Maximum number of samples to visualize')
    parser.add_argument('--framerate', type=int, default=10, help='Frame rate for output video')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Load NuScenes dataset
    global nusc
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    
    # Load inference results
    results = mmcv.load(args.result_path)
    
    # Create output directories
    frames_dir = os.path.join(args.save_path, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Get sample tokens
    sample_tokens = list(results['results'].keys())
    if args.max_samples > 0 and len(sample_tokens) > args.max_samples:
        sample_tokens = sample_tokens[:args.max_samples]
    
    # Visualize each sample
    for i, sample_token in enumerate(tqdm(sample_tokens, desc="Visualizing samples")):
        # Extract trajectories from results
        sample_result = results['results'][sample_token]
        
        # Prepare data for visualization
        visualization_data = {
            'trajectories': {}
        }
        
        # Extract trajectory information if available
        if 'trajectories' in sample_result:
            visualization_data['trajectories'] = sample_result['trajectories']
        
        # Visualize the sample
        frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
        visualize_law_predictions(sample_token, visualization_data, out_path=frames_dir)
        
        # Rename the output file to match the frame pattern
        sample_vis_path = os.path.join(frames_dir, f"{sample_token}_visualization.png")
        if os.path.exists(sample_vis_path):
            os.rename(sample_vis_path, frame_path)
    
    # Create video from frames
    video_path = os.path.join(args.save_path, "law_visualization.mp4")
    create_video_from_frames(frames_dir, video_path, args.framerate)
    print(f"Visualization video saved to: {video_path}")

if __name__ == '__main__':
    main()
