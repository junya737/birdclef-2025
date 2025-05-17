import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from pathlib import Path
# module/models_lib.py
from module.model import BirdCLEFModel  # ✅ 絶対インポートに変更


def find_model_files(cfg):
    """
    指定されたmodel_path以下の.pthファイルをすべて探す
    """
    model_files = []
    model_dir = Path(cfg.model_path)
    for path in model_dir.glob('**/*.pth'):
        model_files.append(str(path))
    return model_files

def load_models(cfg):
    """
    モデルファイルをロードして、推論用に準備する
    - モデルアーキテクチャは BirdCLEFModel を使う
    - 指定foldのみロードしたい場合はcfg.foldsを使用
    """
    
    models = []
    model_files = find_model_files(cfg)

    if not model_files:
        print(f"Warning: No model files found under {cfg.model_path}!")
        return models

    print(f"Found {len(model_files)} model files.")

    if getattr(cfg, "use_specific_folds", False):
        # 指定foldのみフィルタリング
        filtered_files = []
        for fold in cfg.folds:
            fold_files = [f for f in model_files if f"fold{fold}" in f]
            filtered_files.extend(fold_files)
        model_files = filtered_files
        print(f"Using {len(model_files)} model files for the specified folds: {cfg.folds}")

    for model_path in model_files:
        try:
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))

            model = BirdCLEFModel(cfg)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()  # 推論モードにする

            models.append(model)
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")

    return models


