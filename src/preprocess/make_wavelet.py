import os
import numpy as np
import argparse
from scipy import signal
from pathlib import Path
from tqdm import tqdm
import mne
import pywt
from skimage.transform import resize
import matplotlib.pyplot as plt

# --- 설정 ---
FS = 250  # 샘플링 주파수
TARGET_FREQ = 32
TARGET_TIME = 32

# 표준 EEG 채널명 (22채널)
EEG_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]

# GDF 파일에서 읽히는 실제 채널명 (BCI Competition IV 2a 공식 매핑)
BCI2A_MONTAGE = [
    'EEG-Fz', 'EEG-FC3', 'EEG-FC1', 'EEG-FCz', 'EEG-FC2', 'EEG-FC4',
    'EEG-C5', 'EEG-C3', 'EEG-C1', 'EEG-Cz', 'EEG-C2', 'EEG-C4', 'EEG-C6',
    'EEG-CP3', 'EEG-CP1', 'EEG-CPz', 'EEG-CP2', 'EEG-CP4',
    'EEG-P1', 'EEG-Pz', 'EEG-P2', 'EEG-POz'
]

# --- 웨이블릿 변환 함수 ---
def compute_wavelet(eeg_data, fs=FS, target_shape=(TARGET_FREQ, TARGET_TIME)):
    specs = []
    for channel_data in eeg_data:
        b, a = signal.butter(4, [4, 40], btype='bandpass', fs=fs)
        filtered = signal.filtfilt(b, a, channel_data)
        scales = pywt.central_frequency('morl') * fs / (2 * np.linspace(4, 40, target_shape[0]))
        cwt_matrix, _ = pywt.cwt(filtered, scales, 'morl')
        cwt_abs = np.abs(cwt_matrix)
        spec = resize(cwt_abs, target_shape, mode='reflect', anti_aliasing=True)
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
        specs.append(spec)
    specs = np.stack(specs, axis=0)
    return specs

# --- 샘플 시각화 함수 ---
def plot_sample(spec, save_path, title=None):
    n_ch = spec.shape[0]
    ncols = 6
    nrows = int(np.ceil(n_ch / ncols))
    plt.figure(figsize=(ncols * 2, nrows * 2))
    for ch in range(n_ch):
        plt.subplot(nrows, ncols, ch + 1)
        plt.imshow(spec[ch], aspect='auto', cmap='jet')
        plt.title(f'Ch {ch}')
        plt.axis('off')
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- main ---
def main(args):
    raw_dir = Path(args.raw_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])

    for subject in tqdm(subjects, desc="Subjects"):
        session_type = 'T'  # T 세션만 사용
        gdf_files = list((raw_dir / subject).glob(f"*{session_type}.gdf"))
        if not gdf_files:
            continue
        gdf_file = gdf_files[0]

        # GDF 파일 불러오기
        raw = mne.io.read_raw_gdf(str(gdf_file), preload=True)

        # EEG 채널만 선택, 채널명 표준화
        raw.pick_types(eeg=True, eog=False)
        print(f"{subject} {session_type} 채널명:", raw.ch_names)
        # 실제 GDF 채널명 (subject01 기준, 22채널만)
        gdf_names = [
            'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3',
            'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11',
            'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16'
        ]
        # 표준 EEG 채널명
        EEG_CHANNELS = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
        ]
        rename_dict = {old: new for old, new in zip(gdf_names, EEG_CHANNELS)}
        raw.rename_channels(rename_dict)
        raw.pick_channels(EEG_CHANNELS)
        raw.filter(l_freq=0.5, h_freq=100)

        annotations = raw.annotations

        labels_map = {
            '769': 1,  # left_hand
            '770': 2,  # right_hand
            '771': 3,  # feet
            '772': 4,  # tongue
        }

        trial_starts = []
        trial_labels = []

        for desc, onset in zip(annotations.description, annotations.onset):
            desc_str = str(desc)
            if desc_str in labels_map:
                start = int(onset * FS)
                trial_starts.append(start)
                trial_labels.append(labels_map[desc_str])

        if len(trial_starts) == 0:
            print(f"Warning: {subject} {session_type} - 운동 상상 이벤트 없음")
            continue

        trial_len = int(FS * 4)
        data = raw.get_data()

        sample_saved = False

        for idx, (start, label) in enumerate(zip(trial_starts, trial_labels)):
            trial_data = data[:, start : start + trial_len]
            if trial_data.shape[1] < trial_len or np.isnan(trial_data).any():
                continue
            spec = compute_wavelet(trial_data)

            save_path = save_dir / subject / session_type
            save_path.mkdir(parents=True, exist_ok=True)

            filename = f"{subject}_{session_type}_class{label}_trial{idx+1:03d}.npy"
            np.save(save_path / filename, spec)
       
            # 샘플 시각화 (subject마다 첫 trial만)
            if not sample_saved:
                plot_sample(
                    spec,
                    save_path / f"{subject}_{session_type}_class{label}_trial{idx+1:03d}_viz.png",
                    title=f"{subject} {session_type} class{label} trial{idx+1}"
                )
                sample_saved = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()

    main(args)
