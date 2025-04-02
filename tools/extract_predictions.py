#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
import torch
import argparse
from tqdm import tqdm
from mmcv import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Extract predictions from LAW model results')
    parser.add_argument('config', help='Config file')
    parser.add_argument('--result', required=True, help='Result pkl file from inference')
    parser.add_argument('--output', required=True, help='Output file path for extracted predictions')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 設定ファイルの読み込み
    cfg = Config.fromfile(args.config)
    
    # 予測結果の読み込み
    print(f"\n=== Loading prediction results from {args.result} ===")
    try:
        with open(args.result, 'rb') as f:
            result_data = pickle.load(f)
            
        print("\n=== Result Data Analysis ===")
        print(f"Type of result_data: {type(result_data)}")
        
        if isinstance(result_data, list):
            print(f"\nList length: {len(result_data)}")
            if len(result_data) > 0:
                print("\nAnalyzing first item:")
                first_item = result_data[0]
                print(f"Type of first item: {type(first_item)}")
                if isinstance(first_item, dict):
                    print("\nKeys in first item:")
                    for key, value in first_item.items():
                        print(f"\nKey: {key}")
                        print(f"Type: {type(value)}")
                        if isinstance(value, (np.ndarray, torch.Tensor)):
                            print(f"Shape: {value.shape}")
                        elif isinstance(value, list):
                            print(f"Length: {len(value)}")
                            if len(value) > 0:
                                print(f"First element type: {type(value[0])}")
                        elif isinstance(value, dict):
                            print(f"Dict keys: {list(value.keys())}")
                
                if len(result_data) > 1:
                    print("\nAnalyzing second item:")
                    second_item = result_data[1]
                    print(f"Type of second item: {type(second_item)}")
                    if isinstance(second_item, dict):
                        print("Keys in second item:", list(second_item.keys()))
        
        elif isinstance(result_data, dict):
            print("\nDict keys:", list(result_data.keys()))
            for key, value in result_data.items():
                print(f"\nKey: {key}")
                print(f"Type: {type(value)}")
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    print(f"Shape: {value.shape}")
                elif isinstance(value, list):
                    print(f"Length: {len(value)}")
                    if len(value) > 0:
                        print(f"First element type: {type(value[0])}")
                elif isinstance(value, dict):
                    print(f"Dict keys: {list(value.keys())}")
        
        print("\n=== End of Analysis ===\n")
        
    except FileNotFoundError:
        print(f"Error: Result file {args.result} not found!")
        return
    except Exception as e:
        print(f"Error loading result file: {str(e)}")
        return

    # 出力用辞書の初期化
    extracted_results = {}
    trajectories = {}
    
    if isinstance(result_data, list):
        print(f"Found {len(result_data)} prediction results")
        
        for sample_idx, sample_result in enumerate(tqdm(result_data, desc="Processing results")):
            # メタデータの抽出
            meta_info = None
            if isinstance(sample_result, dict):
                if 'img_metas' in sample_result:
                    meta_info = sample_result['img_metas']
                elif 'img_meta' in sample_result:
                    meta_info = sample_result['img_meta']
                
                if meta_info is None:
                    print(f"Warning: No metadata found in sample {sample_idx}")
                    print(f"Available keys in sample: {list(sample_result.keys())}")
                    if 'boxes_3d' in sample_result:
                        print(f"boxes_3d type: {type(sample_result['boxes_3d'])}")
                        if hasattr(sample_result['boxes_3d'], 'shape'):
                            print(f"boxes_3d shape: {sample_result['boxes_3d'].shape}")
                    continue
            else:
                print(f"Warning: Sample {sample_idx} is not a dictionary, type: {type(sample_result)}")
                continue
            
            # サンプルトークンの抽出
            sample_token = None
            if isinstance(meta_info, list) and len(meta_info) > 0:
                sample_token = meta_info[0].get('sample_idx', f"sample_{sample_idx}")
            elif isinstance(meta_info, dict):
                sample_token = meta_info.get('sample_idx', f"sample_{sample_idx}")
            
            if sample_token is None:
                print(f"Warning: No sample token found in sample {sample_idx}")
                print(f"Meta info type: {type(meta_info)}")
                if isinstance(meta_info, list):
                    print(f"Meta info length: {len(meta_info)}")
                    if len(meta_info) > 0:
                        print(f"First meta info keys: {list(meta_info[0].keys())}")
                continue
            
            # 物体検出結果の抽出
            boxes = []
            if 'boxes_3d' in sample_result:
                detection_boxes = sample_result['boxes_3d']
                detection_scores = sample_result.get('scores_3d', None)
                detection_labels = sample_result.get('labels_3d', None)
                
                if detection_boxes is not None:
                    for i, box in enumerate(detection_boxes):
                        box_dict = {
                            'translation': box[:3].tolist() if isinstance(box, np.ndarray) else box[:3],
                            'size': box[3:6].tolist() if isinstance(box, np.ndarray) else box[3:6],
                            'rotation': box[6:].tolist() if isinstance(box, np.ndarray) else box[6:],
                            'name': 'car',  # デフォルトカテゴリ
                            'score': detection_scores[i] if detection_scores is not None else 1.0
                        }
                        
                        # クラスラベルがあれば設定
                        if detection_labels is not None:
                            label_idx = detection_labels[i]
                            # クラスマッピングが設定ファイルにあれば使用
                            if hasattr(cfg, 'class_names'):
                                box_dict['name'] = cfg.class_names[label_idx]
                            else:
                                # デフォルトのクラス名マッピング
                                class_map = {
                                    0: 'car', 1: 'truck', 2: 'bus', 3: 'trailer',
                                    4: 'construction_vehicle', 5: 'pedestrian',
                                    6: 'motorcycle', 7: 'bicycle', 8: 'traffic_cone',
                                    9: 'barrier'
                                }
                                box_dict['name'] = class_map.get(label_idx, 'unknown')
                        
                        boxes.append(box_dict)
            
            # 軌跡結果の抽出
            if 'trajectories' in sample_result:
                traj_data = sample_result['trajectories']
                # GT軌跡
                gt_traj = None
                if 'gt' in traj_data:
                    gt_traj = traj_data['gt']
                    if torch.is_tensor(gt_traj):
                        gt_traj = gt_traj.cpu().numpy()
                
                # 予測軌跡
                pred_traj = None
                if 'pred' in traj_data:
                    pred_traj = traj_data['pred']
                    if torch.is_tensor(pred_traj):
                        pred_traj = pred_traj.cpu().numpy()
                
                trajectories[sample_token] = {
                    'gt': gt_traj,
                    'pred': pred_traj
                }
            
            # 結果の保存
            extracted_results[sample_token] = boxes
    
    # 軌跡情報を結果に追加
    extracted_results['trajectories'] = trajectories
    
    # 結果の保存
    print(f"Saving extracted predictions to {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(extracted_results, f)
    
    print("Extraction complete!")

if __name__ == '__main__':
    main() 