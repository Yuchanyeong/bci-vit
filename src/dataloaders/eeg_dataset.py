# vit/src/dataloaders/eeg_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGSpectrogramDataset(Dataset):
    def __init__(self, data_dir, subjects=None, session_types=['T', 'E'], transform=None, normalize=False):
        """
        data_dir: 전처리된 데이터 루트 경로 (vit/data/preprocessed)
        subjects: 사용할 subject 리스트 (ex: ['subject01', 'subject02', ...])
        session_types: 사용할 세션 (기본은 T와 E 모두)
        transform: torchvision-style transform 적용 가능
        """
        self.samples = []
        self.labels = []
        self.transform = transform
        self.normalize = normalize  # 정규화 여부를 받는 파라미터

        subjects = sorted(os.listdir(data_dir)) if subjects is None else subjects

        for subject in subjects:
            for session in session_types:
                folder = os.path.join(data_dir, subject, session)
                if not os.path.exists(folder):
                    continue
                for fname in os.listdir(folder):
                    if fname.endswith('.npy'):
                        fpath = os.path.join(folder, fname)
                        label = int(fname.split('_')[2].replace('class', '')) - 1  # 0-indexed 레이블
                        self.samples.append(fpath)
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        data = np.load(path)
        
        
        # Z-score 정규화
        if self.normalize:
            mean = np.mean(data, axis=(1, 2), keepdims=True)  # 채널마다 평균
            std = np.std(data, axis=(1, 2), keepdims=True)  # 채널마다 표준편차
            data = (data - mean) / (std + 1e-7)  # 나누기 0을 방지하기 위해 작은 값 추가

        data = torch.tensor(data, dtype=torch.float32)
        
        if self.transform:
            data = self.transform(data)
        
        return data, label
