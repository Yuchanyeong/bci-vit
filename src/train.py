# vit/src/train.py

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

from src.dataloaders.eeg_dataset import EEGSpectrogramDataset
from src.models.vit_custom import create_vit_model, CNN_ViT_Model

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    preprocessed_dir = config['data']['preprocessed_dir']
    test_subject = config['data']['test_subject']

    # 학습/테스트 데이터 준비
    subjects = sorted(os.listdir(preprocessed_dir))
    train_subjects = [s for s in subjects if s != test_subject]
    test_subjects = [test_subject]
    
    normalize = True

    train_dataset = EEGSpectrogramDataset(preprocessed_dir, subjects=train_subjects)
    test_dataset  = EEGSpectrogramDataset(preprocessed_dir, subjects=test_subjects)

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=0)

    model_type = config['model'].get('type', 'vit')
    # 모델 생성
    if model_type == 'vit':
        model = create_vit_model(
            img_size=config['model']['img_size'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            in_chans=config['model']['in_chans'],
        ).to(device)
    elif model_type == 'cnn_vit':
        model = CNN_ViT_Model(
           img_size=config['model']['img_size'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            in_chans=config['model']['in_chans'],
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['learning_rate'],weight_decay=0.01)

    best_acc = 0.0
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, config['train']['num_epochs'] + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Train Accuracy: {train_acc*100:.2f}%")
        
        train_losses.append(avg_loss)
       
        train_accuracies.append(train_acc)
        
        # 평가
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
        # train loop 끝에서
        
        test_accuracies.append(test_acc)
        
        # 최고 성능 모델 저장
        exp_dir = Path(config['data']['save_experiment_dir'])
        exp_dir.mkdir(parents=True, exist_ok=True)
        if test_acc > best_acc:
            torch.save(model.state_dict(), exp_dir / 'best_model.pth')
            best_acc = test_acc

    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    
     # --- 학습 곡선 시각화 및 저장 ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([x*100 for x in train_accuracies], label='Train Acc')
    plt.plot([x*100 for x in test_accuracies], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(exp_dir / 'train_curves.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Config 파일 경로")
    args = parser.parse_args()

    main(args)

