import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def visualize_subject(subject_dir, save_dir, channel_idx=0):
    subject_dir = Path(subject_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for session in ['T', 'E']:
        session_path = subject_dir / session
        if not session_path.exists():
            continue

        files = sorted(session_path.glob("*.npy"))
        for file in tqdm(files, desc=f"{subject_dir.name}_{session}"):
            # 파일 이름 파싱
            parts = file.stem.split('_')
            class_label = parts[2].replace('class', '')  # class1 → 1
            trial_num = parts[3].replace('trial', '')    # trial003 → 003

            # 저장 디렉토리: class별로 구분
            class_save_dir = save_dir / f"class{class_label}"
            class_save_dir.mkdir(parents=True, exist_ok=True)

            # 로드 및 시각화
            spec = np.load(file)  # (25, 32, 32)
            img = spec[channel_idx]  # 선택한 채널

            plt.figure(figsize=(4, 4))
            plt.imshow(img, aspect='auto', origin='lower', cmap='magma')
            plt.colorbar()
            plt.title(f"{file.stem} (ch {channel_idx})")
            plt.tight_layout()

            save_path = class_save_dir / f"{file.stem}_ch{channel_idx}.png"
            plt.savefig(save_path)
            plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_dir", type=str, required=True, help="e.g., vit/data/preprocessed/subject03")
    parser.add_argument("--save_dir", type=str, required=True, help="e.g., vit/visualizations/subject03")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to visualize (default=0)")
    args = parser.parse_args()

    visualize_subject(args.subject_dir, args.save_dir, args.channel)
