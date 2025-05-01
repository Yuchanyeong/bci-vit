import os
import numpy as np
import argparse
from scipy import signal
from pathlib import Path
from tqdm import tqdm
import mne
import pywt
from skimage.transform import resize

# --- 설정 ---
FS = 250  # 샘플링 주파수
TARGET_FREQ = 32
TARGET_TIME = 32

# --- 웨이블릿 변환 함수 ---
def compute_wavelet(eeg_data, fs=FS, target_shape=(TARGET_FREQ, TARGET_TIME)):
    specs = []
    for channel_data in eeg_data:
        # 1. 4-40Hz 대역 통과 필터
        b, a = signal.butter(4, [4, 40], btype='bandpass', fs=fs)
        filtered = signal.filtfilt(b, a, channel_data)
        # 2. 웨이블릿 변환 (모를렛, 32스케일)
        scales = pywt.central_frequency('morl') * fs / (2 * np.linspace(4, 40, target_shape[0]))
        cwt_matrix, _ = pywt.cwt(filtered, scales, 'morl')
        cwt_abs = np.abs(cwt_matrix)
        # 3. 크기 맞추기 (32x32)
        spec = resize(cwt_abs, target_shape, mode='reflect', anti_aliasing=True)
        # 4. 정규화
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
        specs.append(spec)
    specs = np.stack(specs, axis=0)
    return specs

# --- main ---
def main(args):
    raw_dir = Path(args.raw_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])

    for subject in tqdm(subjects, desc="Subjects"):
        for session_type in ['T', 'E']:
            gdf_files = list((raw_dir / subject).glob(f"*{session_type}.gdf"))
            if not gdf_files:
                continue
            gdf_file = gdf_files[0]

            # 세션별로 GDF 파일을 직접 불러오기 (MOABBDataset 대신)
            raw = mne.io.read_raw_gdf(str(gdf_file), preload=True)

            # EEG + EOG 채널만 선택, 필터링
            raw.pick_types(eeg=True, eog=True)
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
            
            print(f"{subject} {session_type} annotation descriptions:", set(annotations.description))

            for desc, onset in zip(annotations.description, annotations.onset):
                if desc in labels_map:
                    start = int(onset * FS)  # 초 -> 샘플
                    trial_starts.append(start)
                    trial_labels.append(labels_map[desc])

            if len(trial_starts) == 0:
                print(f"Warning: {subject} {session_type} - 운동 상상 이벤트 없음")
                continue

            trial_len = int(FS * 4)  # 4초
            data = raw.get_data()  # (채널, 시간)

            for idx, (start, label) in enumerate(zip(trial_starts, trial_labels)):
                trial_data = data[:, start : start + trial_len]
                if trial_data.shape[1] < trial_len:
                    continue
                spec = compute_wavelet(trial_data)

                save_path = save_dir / subject / session_type
                save_path.mkdir(parents=True, exist_ok=True)

                filename = f"{subject}_{session_type}_class{label}_trial{idx+1:03d}.npy"
                np.save(save_path / filename, spec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()

    main(args)
