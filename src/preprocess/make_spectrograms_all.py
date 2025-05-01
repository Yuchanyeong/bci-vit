import os
import numpy as np
import argparse
from scipy import signal
from pathlib import Path
from braindecode.datasets import MOABBDataset
from tqdm import tqdm
import mne

# --- 설정 ---
FS = 250
N_PER_SEG = 64
N_OVERLAP = 32
TARGET_FREQ = 32
TARGET_TIME = 32

# --- 스펙트로그램 변환 ---
def compute_spectrogram(eeg_data, fs=FS):
    specs = []
    for channel_data in eeg_data:
        f, t, Sxx = signal.spectrogram(channel_data, fs=fs, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
        Sxx = np.log1p(Sxx)
        spec = np.zeros((TARGET_FREQ, TARGET_TIME))
        spec[:min(Sxx.shape[0], TARGET_FREQ), :min(Sxx.shape[1], TARGET_TIME)] = Sxx[:TARGET_FREQ, :TARGET_TIME]
        specs.append(spec)
    return np.stack(specs, axis=0)

# --- 메인 ---
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

            subj_num = int(subject.replace("subject", ""))
            dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subj_num])
            raw = dataset.datasets[0].raw

            raw.pick_types(eeg=True, eog=True)
            raw.filter(l_freq=0.5, h_freq=100)

            events, event_ids = mne.events_from_annotations(raw)

            labels_map = {
               'left_hand': 1,
                'right_hand': 2,
                'feet': 3,
                'tongue': 4,
            }

            trial_len = int(FS * 4)  # 4초 trial
            data = raw.get_data()

            trial_starts = []
            trial_labels = []

            for event in events:
                onset_sample = event[0]
                code = event[2]
                if code in labels_map:
                    trial_starts.append(onset_sample)
                    trial_labels.append(labels_map[code])

            if len(trial_starts) == 0:
                print(f"Warning: {subject} {session_type} - 운동 상상 이벤트 없음")
                continue

            for idx, (start, label) in enumerate(zip(trial_starts, trial_labels)):
                trial_data = data[:, start : start + trial_len]
                if trial_data.shape[1] < trial_len:
                    continue
                spec = compute_spectrogram(trial_data)

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
