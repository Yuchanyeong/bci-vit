import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

# 시각화할 npy 파일 경로 리스트 만들기
npy_files = sorted(glob.glob('data/preprocessed_wave/subject01/T/subject01_T_class1_trial*.npy'))

# 저장할 디렉토리 지정
save_dir = Path('visualization_wave/subject01/T/class1')
save_dir.mkdir(parents=True, exist_ok=True)

# 시각화할 trial 개수 지정 (예시: 20개)
num_to_show = 20
files_to_show = npy_files[:num_to_show]

ncols = 4
nrows = int(np.ceil(num_to_show / ncols))
plt.figure(figsize=(ncols*4, nrows*4))
for idx, f in enumerate(files_to_show):
    spec = np.load(f)
    img = spec[1]
    plt.subplot(nrows, ncols, idx+1)
    plt.imshow(img, aspect='auto', cmap='jet')
    plt.title(os.path.basename(f), fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_dir / 'trials_grid1.png', bbox_inches='tight')
plt.close()
print(f"{len(files_to_show)} trial 이미지가 {save_dir}에 저장되었습니다.")
