# vit/src/models/vit_custom.py

import timm
import torch
import torch.nn as nn


def create_vit_model(img_size=32, patch_size=4, num_classes=4, in_chans=25, dropout_rate=0.5):
    """
    ViT 모델 생성: EEG 입력에 맞게 Conv2D 수정 + 입력 크기 재정의 + 드롭아웃 적용
    """
    model = timm.create_model(
        'vit_tiny_patch16_224',  # 기본 구조는 가져오되 수정함
        pretrained=True,
        num_classes=num_classes,
        drop_rate=dropout_rate,     # 전체 dropout
        attn_drop_rate=dropout_rate # attention dropout
    )

    # patch embedding 수정
    model.patch_embed.proj = nn.Conv2d(
        in_channels=in_chans,
        out_channels=model.patch_embed.proj.out_channels,
        kernel_size=(patch_size, patch_size),
        stride=(patch_size, patch_size)
    )

    # 내부 이미지 크기 하드코딩 해제
    model.patch_embed.img_size = (img_size, img_size)
    model.patch_embed.grid_size = (img_size // patch_size, img_size // patch_size)
    model.num_patches = model.patch_embed.grid_size[0] * model.patch_embed.grid_size[1]

    # Position embedding 크기도 재정의 (임베딩 수가 바뀌니까)
    model.pos_embed = nn.Parameter(
        torch.zeros(1, model.num_patches + 1, model.pos_embed.shape[-1])
    )
    nn.init.trunc_normal_(model.pos_embed, std=0.02)

    # 필요하다면 classification head에 dropout 추가
    # (timm ViT는 이미 drop_rate로 head에 dropout 적용함)
    # 만약 더 강하게 적용하고 싶으면 아래처럼 커스텀 head를 추가할 수 있음:
    # model.head = nn.Sequential(
    #     nn.Dropout(dropout_rate),
    #     nn.Linear(model.head.in_features, num_classes)
    # )

    return model


class CNN_ViT_Model(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=4, in_chans=25, dropout_rate=0.5):
        super(CNN_ViT_Model, self).__init__()

        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(1, 64)  # GroupNorm(num_groups=1, num_channels=64)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(1, 128)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.gn3 = nn.GroupNorm(1, 256)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        self.vit = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=True,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            in_chans=256
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.gn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.pool3(x)

        x = self.resize(x)
        x = self.vit(x)
        return x