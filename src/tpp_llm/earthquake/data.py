"""
Dataset and Data Loader for the TPP-LLM
"""
import json
import os
import random
import shutil
import torch
import numpy as np
from torch.utils.data import Dataset

#torch.manual_seed(0)

class TPPLLMDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def float32_to_bytes_big_endian(value):
    bytes_obj = np.array([value], dtype=np.float32).tobytes()
    return bytes_obj[3], bytes_obj[2], bytes_obj[1], bytes_obj[0]

def float32_to_byte_tokens(value):
    int1, int2, int3, int4 = float32_to_bytes_big_endian(value)
    return f"<|byte_{int1}|><|byte_{int2}|><|byte_{int3}|><|byte_{int4}|>"

def float32_list_to_byte_token_list(values):
    return [float32_to_byte_tokens(v) for v in values]

def get_collate_fn(stats=None):
    """
    stats（統計量）を受け取って、標準化機能付きの collate_fn を返すファクトリ関数
    """
    def collate_fn(batch: dict) -> dict:
        # 返却用の辞書を準備
        collated = {
            'time_since_start': [torch.FloatTensor(item['time_since_start']) for item in batch],
            'time_since_last_event': [torch.FloatTensor(item['time_since_last_event']) for item in batch],
            'type_event': [torch.LongTensor(item['type_event']) for item in batch],
            'type_text': [item['type_text'] for item in batch],
            'magnitude' : [torch.FloatTensor(item['magnitude']) for item in batch],
            'depth': [torch.FloatTensor(item['depth']) for item in batch],
            'latitude' : [torch.FloatTensor(item['latitude']) for item in batch],
            'longitude': [torch.FloatTensor(item['longitude']) for item in batch],
        }

        # stats が渡されている場合は、標準化を行ってからバイトトークン化する
        if stats is not None:
            time_mean, time_std = stats['time']['mean'], stats['time']['std']
            mag_mean, mag_std = stats['mag']['mean'], stats['mag']['std']
            dep_mean, dep_std = stats['dep']['mean'], stats['dep']['std']
            lat_mean, lat_std = stats['lat']['mean'], stats['lat']['std']
            lon_mean, lon_std = stats['lon']['mean'], stats['lon']['std']
            
            time_tokens = []
            mag_tokens = []
            dep_tokens = []
            lat_tokens = []
            lon_tokens = []

            for item in batch:
                # 1. 生の数値を取得
                raw_time = np.array(item['time_since_start'], dtype=np.float32)
                raw_mag = np.array(item['magnitude'], dtype=np.float32)
                raw_dep = np.array(item['depth'], dtype=np.float32)
                raw_lat = np.array(item['latitude'], dtype=np.float32)
                raw_lon = np.array(item['longitude'], dtype=np.float32)

                # 2. 標準化（ゼロ除算防止に 1e-8 を足す）
                norm_time = (raw_time - time_mean) / (time_std + 1e-8)
                norm_mag = (raw_mag - mag_mean) / (mag_std + 1e-8)
                norm_dep = (raw_dep - dep_mean) / (dep_std + 1e-8)
                norm_lat = (raw_lat - lat_mean) / (lat_std + 1e-8)
                norm_lon = (raw_lon - lon_mean) / (lon_std + 1e-8)

                # 3. 標準化された数値をバイトトークンリストに変換
                time_tokens.append(float32_list_to_byte_token_list(norm_time))
                mag_tokens.append(float32_list_to_byte_token_list(norm_mag))
                dep_tokens.append(float32_list_to_byte_token_list(norm_dep))
                lat_tokens.append(float32_list_to_byte_token_list(norm_lat))
                lon_tokens.append(float32_list_to_byte_token_list(norm_lon))

            collated['time_float'] = time_tokens       # 標準化済みのバイトトークン文字列
            collated['magnitude_float'] = mag_tokens  # 標準化済みのバイトトークン文字列
            collated['depth_float'] = dep_tokens      # 標準化済みのバイトトークン文字列
            collated['latitude_float'] = lat_tokens  # 標準化済みのバイトトークン文字列
            collated['longitude_float'] = lon_tokens      # 標準化済みのバイトトークン文字列
            
        return collated

    return collate_fn

def create_few_shot_dataset(data_dir, output_dir, few_shot_ratio=0.1, seed=0) -> None:
    random.seed(seed)
    train_file = os.path.join(data_dir, 'train.json')
    dev_file = os.path.join(data_dir, 'dev.json')
    test_file = os.path.join(data_dir, 'test.json')
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    few_shot_size = int(len(train_data) * few_shot_ratio)
    few_shot_train_data = random.sample(train_data, few_shot_size)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train.json'), 'w') as file:
        json.dump(few_shot_train_data, file, indent=4)
    shutil.copy(dev_file, os.path.join(output_dir, 'dev.json'))
    shutil.copy(test_file, os.path.join(output_dir, 'test.json'))
    print(f"Few-shot dataset created in {output_dir}.")

def get_all_data_stats(dataset_dir):
    """
    Reads all Train files and calculates the global 
    'maximum', 'minimum', 'mean', and 'std' values across the entire dataset.
    """
    print(f"Calculating global stats (min/max/mean/std) from train files in {dataset_dir}...")

    stats = {
        'mag': {'min': float('inf'), 'max': float('-inf'), 'mean': 0.0, 'std': 1.0},
        'dep': {'min': float('inf'), 'max': float('-inf'), 'mean': 0.0, 'std': 1.0},
        'time': {'min': float('inf'), 'max': float('-inf'), 'mean': 0.0, 'std': 1.0},
        'delta': {'min': float('inf'), 'max': float('-inf'), 'mean': 0.0, 'std': 1.0},
        'lat': {'min': float('inf'), 'max': float('-inf'), 'mean': 0.0, 'std': 1.0},
        'lon': {'min': float('inf'), 'max': float('-inf'), 'mean': 0.0, 'std': 1.0}
        
    }
    # クラスごとのマグニチュードを収集
    mags_by_class = {k: [] for k in range(3)}

    all_mag = []
    all_dep = []
    all_time = []
    all_delta = []
    all_lat = []
    all_lon = []

    target_files = ['train.json'] 

    for fname in target_files:
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            print(f"  - {fname} not found, skipping.")
            continue
        
        print(f"  - Scanning {fname}...")
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            for seq in data:
                m_list = seq.get('magnitude', [])
                type_list = seq.get('type_event', [])

                if m_list and type_list:
                    if len(m_list) != len(type_list):
                        print("Warning: magnitude と type_event の長さが不一致")
                    else:
                        for mag, event_type in zip(m_list, type_list):
                            if event_type in mags_by_class and np.isfinite(mag):
                                mags_by_class[event_type].append(float(mag))
                if m_list:
                    all_mag.extend(m_list) # リストにデータを追加
                    stats['mag']['min'] = min(stats['mag']['min'], min(m_list))
                    stats['mag']['max'] = max(stats['mag']['max'], max(m_list))
                
                d_list = seq.get('depth', [])
                if d_list:
                    all_dep.extend(d_list)
                    stats['dep']['min'] = min(stats['dep']['min'], min(d_list))
                    stats['dep']['max'] = max(stats['dep']['max'], max(d_list))
                
                t_list = seq.get('time_since_start', [])
                if t_list:
                    all_time.extend(t_list)
                    stats['time']['min'] = min(stats['time']['min'], min(t_list)) 
                    stats['time']['max'] = max(stats['time']['max'], max(t_list))

                dt_list = seq.get('time_since_last_event', [])
                if dt_list:
                    all_delta.extend(dt_list)
                    stats['delta']['min'] = min(stats['delta']['min'], min(dt_list))
                    stats['delta']['max'] = max(stats['delta']['max'], max(dt_list))

                lat_list = seq.get('latitude', [])
                if lat_list:
                    all_lat.extend(lat_list)
                    stats['lat']['min'] = min(stats['lat']['min'], min(lat_list)) 
                    stats['lat']['max'] = max(stats['lat']['max'], max(lat_list))

                lon_list = seq.get('longitude', [])
                if lon_list:
                    all_lon.extend(lon_list)
                    stats['lon']['min'] = min(stats['lon']['min'], min(lon_list))
                    stats['lon']['max'] = max(stats['lon']['max'], max(lon_list))
                
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # ===== 全データをまとめた後で平均と標準偏差を計算 =====
    if all_mag:
        stats['mag']['mean'] = float(np.mean(all_mag))
        stats['mag']['std'] = float(np.std(all_mag))
    else:
        stats['mag'] = {'min': -2.0, 'max': 10.0, 'mean': 0.0, 'std': 1.0}

    if all_dep:
        stats['dep']['mean'] = float(np.mean(all_dep))
        stats['dep']['std'] = float(np.std(all_dep))
    else:
        stats['dep'] = {'min': 0.0, 'max': 700.0, 'mean': 0.0, 'std': 1.0}

    if all_time:
        stats['time']['mean'] = float(np.mean(all_time))
        stats['time']['std'] = float(np.std(all_time))
    else:
        stats['time'] = {'min': 0.0, 'max': 100.0, 'mean': 0.0, 'std': 1.0}

    if all_delta:
        stats['delta']['mean'] = float(np.mean(all_delta))
        stats['delta']['std'] = float(np.std(all_delta))
    else:
        stats['delta'] = {'min': 0.0, 'max': 100.0, 'mean': 0.0, 'std': 1.0}

    if all_lat:
        stats['lat']['mean'] = float(np.mean(all_lat))
        stats['lat']['std'] = float(np.std(all_lat))
    else:
        stats['lat'] = {'min': 0.0, 'max': 100.0, 'mean': 0.0, 'std': 1.0}

    if all_lon:
        stats['lon']['mean'] = float(np.mean(all_lon))
        stats['lon']['std'] = float(np.std(all_lon))
    else:
        stats['lon'] = {'min': 0.0, 'max': 100.0, 'mean': 0.0, 'std': 1.0}
    # ========================================================

    print(f"Global Stats Result:")
    print(f"  > Magnitude: {stats['mag']['min']} ~ {stats['mag']['max']} (Mean: {stats['mag']['mean']:.4f}, Std: {stats['mag']['std']:.4f})")
    print(f"  > Depth:     {stats['dep']['min']} ~ {stats['dep']['max']} (Mean: {stats['dep']['mean']:.4f}, Std: {stats['dep']['std']:.4f})")
    print(f"  > latitude: {stats['lat']['min']} ~ {stats['lat']['max']} (Mean: {stats['lat']['mean']:.4f}, Std: {stats['lat']['std']:.4f})")
    print(f"  > longitude:     {stats['lon']['min']} ~ {stats['lon']['max']} (Mean: {stats['lon']['mean']:.4f}, Std: {stats['lon']['std']:.4f})")
    print(f"  > Time:      {stats['time']['min']} ~ {stats['time']['max']} (Mean: {stats['time']['mean']:.4f}, Std: {stats['time']['std']:.4f})")
    print(f"  > Delta:     {stats['delta']['min']} ~ {stats['delta']['max']} (Mean: {stats['delta']['mean']:.4f}, Std: {stats['delta']['std']:.4f})")
    # クラスごとのマグニチュード中心を計算
    class_centers = {}

    for k, values in mags_by_class.items():
        if values:
            class_centers[k] = float(np.median(values))
        else:
            class_centers[k] = None

    stats['magnitude_class_centers'] = class_centers
    print("Magnitude class centers:")
    for k, center in class_centers.items():
        print(f"  > Class {k}: {center}")
    return stats

