
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from pathlib import Path


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=getattr(cfg, 'drop_rate', 0.2),
            drop_path_rate=getattr(cfg, 'drop_path_rate', 0.2)
        )
        
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.use_mlp_head = getattr(cfg, 'use_mlp_head', False)
        
        if self.use_mlp_head:
            hidden_dim = getattr(cfg, 'hidden_dim', 512)
            dropout_rate = getattr(cfg, 'classifier_dropout', 0.3)
            self.classifier = nn.Sequential(
                nn.Linear(backbone_out, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, cfg.num_classes)
            )
        else:
            self.classifier = nn.Linear(backbone_out, cfg.num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha

    def forward(self, x, targets=None):
        if self.training and self.mixup_enabled and targets is not None:
            x, targets_a, targets_b, lam = self.mixup_data(x, targets)
        else:
            targets_a = targets_b = lam = None

        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['features']
        
        if len(features.shape) == 4:
            avg_features = self.avgpool(features)
            max_features = self.maxpool(features)
            features = avg_features + max_features
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)

        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits, logits, targets_a, targets_b, lam)
            return logits, loss

        return logits

    def mixup_data(self, x, targets):
        batch_size = x.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        indices = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[indices]
        return mixed_x, targets, targets[indices], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
