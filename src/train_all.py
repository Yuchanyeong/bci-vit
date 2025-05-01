import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from src.dataloaders.eeg_dataset import EEGSpectrogramDataset
from src.models.vit_custom import create_vit_model
from collections import Counter

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    preprocessed_dir = config['data']['preprocessed_dir']
    subjects = sorted(os.listdir(preprocessed_dir))

    # ✅ 세션 구분 기반 데이터셋 구성
    train_dataset = EEGSpectrogramDataset(preprocessed_dir, subjects=subjects, session_types=['T'])
    test_dataset  = EEGSpectrogramDataset(preprocessed_dir, subjects=subjects, session_types=['E'])

    normalize = True
    print(f"\n[Train] Label Dist: {Counter(train_dataset.labels)}")
    print(f"[Test ] Label Dist: {Counter(test_dataset.labels)}")

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=2)

    # ✅ 모델 생성
    model = create_vit_model(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        num_classes=config['model']['num_classes'],
        in_chans=config['model']['in_chans'],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['learning_rate'], weight_decay=0.01)

    best_acc = 0.0
    exp_dir = Path(config['data']['save_experiment_dir'])
    exp_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config['train']['num_epochs'] + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        train_acc = correct / total
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Train Accuracy: {train_acc*100:.2f}%")

        # ✅ 테스트 평가
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total
        print(f"[Epoch {epoch}] Test Accuracy: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            torch.save(model.state_dict(), exp_dir / 'best_model.pth')
            best_acc = test_acc

    print(f"\n✅ Best Test Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    main(args)
