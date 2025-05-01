# vit/src/evaluate.py

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
from pathlib import Path
import yaml

from src.dataloaders.eeg_dataset import EEGSpectrogramDataset
from src.models.vit_custom import create_vit_model

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    print(f"✅ Evaluation Accuracy: {acc*100:.2f}%")
    print("📊 Prediction Distribution:", Counter(all_preds))
    print("📊 Ground Truth Distribution:", Counter(all_labels))

def main(args):
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    test_subject = config["data"]["test_subject"]
    preprocessed_dir = Path(config["data"]["preprocessed_dir"])
    test_dataset = EEGSpectrogramDataset(preprocessed_dir, subjects=[test_subject])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = create_vit_model(
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        num_classes=config["model"]["num_classes"],
        in_chans=config["model"]["in_chans"],
    ).to(device)

    model_path = Path(config["data"]["save_experiment_dir"]) / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"📂 Loaded model from: {model_path}")

    evaluate(model, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args)
