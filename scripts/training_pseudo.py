#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import logging
import random
import gc
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json

import timm
from torch import nn
import torch.nn.functional as F


logging.basicConfig(level=logging.ERROR)


# In[2]:


print("CUDA available:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("Device name:", torch.cuda.get_device_name(0))
print("Tensor device:", torch.tensor([1.0], device="cuda").device)
print(torch.cuda.get_arch_list())


# In[3]:


class BirdCLEFDatasetFromNPY_Mixup(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train", label2idx=None, idx2label=None):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms
        self.label_to_idx = label2idx
        self.idx_to_label = idx2label
        self.species_ids = label2idx.keys() if label2idx else []
        self.num_classes = len(self.species_ids)
        
        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename

        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row1 = self.df.iloc[idx]
        spec1 = self._get_spec(row1['samplename'])
        label1 = self._get_label(row1)

        # === Mixup ===
        if self.mode == "train" and self.cfg.use_mixup and random.random() < self.cfg.mixup_prob:
            idx2 = random.randint(0, len(self.df) - 1)
            row2 = self.df.iloc[idx2]
            spec2 = self._get_spec(row2['samplename'])
            label2 = self._get_label(row2)

            lam = np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha)
            spec = lam * spec1 + (1 - lam) * spec2
            label = lam * label1 + (1 - lam) * label2
        else:
            spec = spec1
            label = label1

        return {
            'melspec': spec,
            'target': torch.tensor(label, dtype=torch.float32),
            'filename': row1['filename']
        }

    def _get_spec(self, samplename):
        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        else:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":
                print(f"Warning: Spectrogram not found: {samplename}")

        spec = torch.tensor(spec, dtype=torch.float32)
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)

        return spec

    def _get_label(self, row):
        target = np.zeros(self.num_classes, dtype=np.float32)
        if row['primary_label'] in self.label_to_idx:
            target[self.label_to_idx[row['primary_label']]] = 1.0

        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']
            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0

        return target

    def apply_spec_augmentations(self, spec):
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0

        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0

        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)

        return spec



class BirdCLEFDatasetWithPseudoMixup(Dataset):
    def __init__(self, df, cfg, spectrograms=None, pseudo_df=None, pseudo_melspecs=None,
                 mode="train", label2idx=None, idx2label=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms
        self.pseudo_df = pseudo_df.reset_index(drop=True) if pseudo_df is not None else None
        self.pseudo_melspecs = pseudo_melspecs
        self.label_to_idx = label2idx
        self.idx2label = idx2label
        self.species_ids = list(label2idx.keys()) if label2idx else []
        self.num_classes = len(self.species_ids)

        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df['filename'].map(
                lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row1 = self.df.iloc[idx]
        spec1 = self._get_spec(row1['samplename'])
        label1 = self._get_label(row1)

        rand_val = random.random()

        # === real × real mixup ===
        if self.mode == "train" and self.cfg.use_mixup and rand_val < self.cfg.mixup_prob:
            idx2 = random.randint(0, len(self.df) - 1)
            row2 = self.df.iloc[idx2]
            spec2 = self._get_spec(row2['samplename'])
            label2 = self._get_label(row2)

            lam = np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha)
            spec = lam * spec1 + (1 - lam) * spec2
            label = lam * label1 + (1 - lam) * label2

            return {
                'melspec': spec,
                'target': torch.tensor(label, dtype=torch.float32),
                'filename': row1['filename']
            }

        # === real × pseudo mixup ===
        if (self.mode == "train" and self.cfg.use_pseudo_mixup and
            self.pseudo_df is not None and
            rand_val < (self.cfg.mixup_prob + self.cfg.pseudo_mixup_prob)):
            
            idx2 = random.randint(0, len(self.pseudo_df) - 1)
            row2 = self.pseudo_df.iloc[idx2]
            spec2 = self._get_spec_pseudo(row2['samplename'])
            label2 = self._get_label_pseudo(row2)

            lam = np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha)
            spec = lam * spec1 + (1 - lam) * spec2
            label = lam * label1 + (1 - lam) * label2

            return {
                'melspec': spec,
                'target': torch.tensor(label, dtype=torch.float32),
                'filename': row1['filename']
            }

        # === no mixup ===
        return {
            'melspec': spec1,
            'target': torch.tensor(label1, dtype=torch.float32),
            'filename': row1['filename']
        }

    def _get_spec(self, samplename):
        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        else:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":
                print(f"Warning: Spectrogram not found: {samplename}")

        spec = torch.tensor(spec, dtype=torch.float32)
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)

        return spec

    def _get_spec_pseudo(self, samplename):
        if self.pseudo_melspecs and samplename in self.pseudo_melspecs:
            spec = self.pseudo_melspecs[samplename]
        else:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":
                print(f"Warning: Pseudo spectrogram not found: {samplename}")

        spec = torch.tensor(spec, dtype=torch.float32)
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)

        return spec  # No augmentation

    def _get_label(self, row):
        target = np.zeros(self.num_classes, dtype=np.float32)
        if row['primary_label'] in self.label_to_idx:
            target[self.label_to_idx[row['primary_label']]] = 1.0

        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']
            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0

        return target

    def _get_label_pseudo(self, row):
        values = row[self.species_ids].values.astype(np.float32)
        values = np.nan_to_num(values, nan=0.0)
        return values

    def apply_spec_augmentations(self, spec):
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0

        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0

        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)

        return spec


# In[4]:


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), 
                            (x.size(-2), x.size(-1))).pow(1. / self.p)

# 差し替え


class BirdCLEFModelForTrain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.2,
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
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        # self.pooling = GeM()
            
        self.feat_dim = backbone_out
        
        # self.dropout = nn.Dropout(0.3)
        # self.activation = nn.Mish()  # または Swish, GELU
        # self.classifier = nn.Sequential(
        #     nn.Linear(backbone_out, backbone_out // 2),
        #     nn.BatchNorm1d(backbone_out // 2),
        #     self.activation,
        #     self.dropout,
        #     nn.Linear(backbone_out // 2, cfg.num_classes)
        # )

        
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)
        # 活性化関数不在．
        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha
            
    def forward(self, x, targets=None):
    
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None
        
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        
        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits, 
                                       logits, targets_a, targets_b, lam)
            return logits, loss
            
        return logits
    
    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        
        return mixed_x, targets, targets[indices], lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    
class BirdCLEFModelForTrain_Coat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # CoaT専用: drop_path_rateを0にする
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.0  # <= ここを0.0に！
        )
        
        # CoaTは reset_classifier が必要
        backbone_out = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0, 'avg')  # <= global_pool='avg'
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)

    def forward(self, x):
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits
    

class BirdCLEFModelForTrain_Swin(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.2
        )
        
        backbone_out = self.backbone.head.in_features
        self.backbone.reset_classifier(0)

        self.pooling = nn.AdaptiveAvgPool2d(1)  # 2Dプーリングに変更！！
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)

    def forward(self, x):
        features = self.backbone(x)

        if isinstance(features, dict):
            features = features['features']

        if features.ndim == 4:
            # CNN系 (B, C, H, W)
            features = self.pooling(features)
            features = features.flatten(1)
        elif features.ndim == 3:
            # Transformer系 (B, N, C)
            features = features.mean(dim=1)
        elif features.ndim == 2:
            # もう (B, C) になってる（例えば SwinTiny）
            pass  # 何も加工しない
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")

        logits = self.classifier(features)
        return logits


# In[ ]:


class CFG:
    def __init__(self, mode="train", kaggle_notebook=False, debug=False):
        assert mode in ["train", "inference"], "mode must be 'train' or 'inference'"
        self.mode = mode
        self.KAGGLE_NOTEBOOK = kaggle_notebook
        self.debug = debug

        # ===== Path Settings =====
        if self.KAGGLE_NOTEBOOK:
            self.OUTPUT_DIR = ''
            self.train_datadir = '/kaggle/input/birdclef-2025/train_audio'
            
            self.test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
            self.submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
            self.taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
            self.model_path = '/kaggle/input/birdclef-2025-0330' 
            self.models_dir = ""
            
            # kaggle notebookならここを変更する．
            # データセットの設定
            self.train_csv = None
            self.spectrogram_npy = None
            
            # Pseudo Labelの設定
            self.pseudo_label_csv = None
            self.pseudo_melspec_npy = None

            
        else:
            self.OUTPUT_DIR = '../data/result/'
            self.RAW_DIR = '../data/raw/'
            self.PROCESSED_DIR = '../data/processed/'
            self.train_datadir = '../data/raw/train_audio/'
            
            self.test_soundscapes = '../data/raw/test_soundscapes/'
            self.submission_csv = '../data/raw/sample_submission.csv'
            self.taxonomy_csv = '../data/raw/taxonomy.csv'
            self.models_dir = "../models/" # 全modelの保存先
            self.model_path = self.models_dir # 各モデルの保存先．学習時に動的に変更．
            
            
            # ローカルならここを変更する
            # データセットの設定
            self.train_csv = '../data/processed/mel_sfzn3_hd_hl16/train.csv'
            self.spectrogram_npy = '../data/processed/mel_sfzn3_hd_hl16//birdclef2025_melspec_5sec_256_256.npy'
            
            # Pseudo Labelの設定
            self.pseudo_label_csv = "../data/processed/pseudo_labels/ensmbl_0850//pseudo_labels.csv"
            self.pseudo_melspec_npy = "../data/processed/mel_prtl_trn_sndscps_hl16_0850/mel_train_soundscapes.npy"


        # ===== Model Settings =====
        self.model_name = "efficientnet_b0" # tf_efficientnetv2_b3    seresnext26t_32x4d eca_nfnet_l0

        self.pretrained = True if mode == "train" else False
        self.in_channels = 1

        # ===== Audio Settings =====
        self.FS = 32000
        self.TARGET_SHAPE = (256, 256)
        
        # trainer内部で決まるのでここでは指定しない．
        self.num_classes = None


        # ===== Training Mode =====
        if mode == "train":
            self.seed = 42
            self.apex = False
            self.print_freq = 100
            self.num_workers = 2

            self.LOAD_DATA = True
            self.epochs = 7
            self.batch_size = 32
            self.criterion = 'BCEWithLogitsLoss'

            self.n_fold = 5
            self.selected_folds = [0, 1, 2, 3, 4] # foldの選択

            self.optimizer = 'AdamW'
            self.lr = 5e-4
            self.weight_decay = 1e-5
            self.scheduler = 'CosineAnnealingLR'
            self.min_lr = 1e-6
            self.T_max = self.epochs
            self.full_train = False
            self.is_RareFull = False # レア種は全部train foldにする
            self.aug_prob = 0.5 # spec augmentの確率
            
            # real × realのmixupの設定
            self.use_mixup = True
            self.mixup_alpha =  0.4
            self.mixup_prob = 0.5
            
            self.secondary_labels = True # secondary_labelsを使うかどうか
            
            
            
            # real × pseudoのmixupの設定
            self.use_pseudo_mixup = True # Pseudo mixupを使うかどうか
            self.pseudo_no_call_threshold = 0.06  # no callの閾値．低いほうがラベルが正確．0.08が上限
            self.pseudo_high_conf_threshold = 0.9 # Pseudo Labelの高信頼度の閾値．高いほうがラベルが正確．0.7が下限．
            self.pseudo_mixup_prob = 0.1 # Pseudo mixup を使う確率．real × real mixupと同時に使われることはない．
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            
            if self.debug:
                self.epochs = 2
                self.selected_folds = [0]
                self.batch_size = 4
                


# In[6]:


cfg = CFG(mode="train", kaggle_notebook=False, debug=False)

if cfg.KAGGLE_NOTEBOOK:
    sys.path.append("/kaggle/input/birdclef-2025-libs/")
from module import  datasets_lib, models_lib, learning_lib, utils_lib


# In[7]:


# trainの処理をクラスで実行．
class BirdCLEFTrainer:
    def __init__(self, cfg, df, taxonomy_df, datasets_lib, models_lib, learning_lib):
        self.cfg = cfg
        self.df = df.head(100).reset_index(drop=True) if cfg.debug else df
        self.taxonomy_df = taxonomy_df
        self.datasets_lib = datasets_lib
        self.models_lib = models_lib
        self.learning_lib = learning_lib
        self.spectrograms = None
        self.pseudo_df = None
        self.pseudo_melspecs = None
        self.best_scores = []
        self.train_metrics = {}
        self.val_metrics = {}
        self.label2index = {}
        self.index2label = {}
        self.num_classes = None

        self._setup_model_dir()
        self._save_config()
        self._build_index_label_mapping()
        self._load_spectrograms()
        
        if self.cfg.use_pseudo_mixup:
            self._load_pseudo_data()

    def _setup_model_dir(self):
        if self.cfg.debug:
            current_time = "debug"
            self.cfg.model_path = os.path.join(self.cfg.models_dir, "models_debug")
        else:
            japan_time = datetime.now(timezone(timedelta(hours=9)))
            current_time = japan_time.strftime('%Y%m%d_%H%M')
            self.cfg.model_path = os.path.join(self.cfg.models_dir, f"models_{current_time}")

        os.makedirs(self.cfg.model_path, exist_ok=True)
        print(f"[INFO] Models will be saved to: {self.cfg.model_path}")

        # dataset-metadata.jsonを保存
        dataset_metadata = {
            "title": f"bc25-models-{current_time}",
            "id": f"ihiratch/bc25-models-{current_time}",
            "licenses": [
                {
                    "name": "CC0-1.0"
                }
            ]
        }
        metadata_path = os.path.join(self.cfg.model_path, "dataset-metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(dataset_metadata, f, indent=2)

    def _save_config(self):
        cfg_dict = vars(self.cfg)
        cfg_df = pd.DataFrame(list(cfg_dict.items()), columns=["key", "value"])
        cfg_df.to_csv(os.path.join(self.cfg.model_path, "config.csv"), index=False)

    def _build_index_label_mapping(self):
        species_ids = self.taxonomy_df['primary_label'].tolist()
        self.cfg.num_classes = len(species_ids)
        # labelとindexの対応
        self.index2label = {i: label for i, label in enumerate(species_ids)}
        self.label2index = {label: i for i, label in enumerate(species_ids)}

        print(self.index2label)

    def _load_spectrograms(self):
        print(f"Loading pre-computed mel spectrograms from NPY file, from the path: {self.cfg.spectrogram_npy}")
        self.spectrograms = np.load(self.cfg.spectrogram_npy, allow_pickle=True).item()
        print(f"Loaded {len(self.spectrograms)} pre-computed mel spectrograms")
        
    def _load_pseudo_data(self):
        print("📥 Loading pseudo label CSV and melspecs from: ", self.cfg.pseudo_label_csv)

        # 1. ラベルCSV読み込み
        df = pd.read_csv(self.cfg.pseudo_label_csv)
        species_cols = df.columns.drop("row_id")
        
        # 2. soft label 前処理: しきい値以下をゼロに
        df[species_cols] = df[species_cols].where(df[species_cols] >= self.cfg.pseudo_no_call_threshold, 0.0)

        # 3. no_call と high_conf に分類
        no_call_df = df[df[species_cols].max(axis=1) == 0.0].copy()
        no_call_df["primary_label"] = "no_call"
        no_call_df["pseudo_source"] = "no_call"
        no_call_df["samplename"] = no_call_df["row_id"]

        high_conf_df = df[df[species_cols].max(axis=1) >= self.cfg.pseudo_high_conf_threshold].copy()
        high_conf_df["primary_label"] = high_conf_df[species_cols].idxmax(axis=1)
        high_conf_df["pseudo_source"] = "high_conf"
        high_conf_df["samplename"] = high_conf_df["row_id"]

        # 4. 統合
        self.pseudo_df = pd.concat([no_call_df, high_conf_df], axis=0).reset_index(drop=True)
        print(f"✅ no_call: {len(no_call_df)}, high_conf: {len(high_conf_df)}, total: {len(self.pseudo_df)}")

        # 5. 必要な row_id だけ抽出
        used_ids = set(self.pseudo_df["row_id"])

        # 6. 辞書形式の .npy を読み込む
        print("📦 Loading full pseudo mel spectrograms from:", self.cfg.pseudo_melspec_npy)
        full_mels = np.load(self.cfg.pseudo_melspec_npy, allow_pickle=True).item()
        
        print(f"📦 All pseudo mel specs loaded: {len(full_mels)}")

        # 7. フィルタリング
        self.pseudo_melspecs = {
            row_id: full_mels[row_id]
            for row_id in used_ids
            if row_id in full_mels
        }
        
        del full_mels  # メモリ節約のために削除
        gc.collect()  # ガーベジコレクションを実行

        print(f"✅ Filtered mel specs loaded: {len(self.pseudo_melspecs)}")
        
    def _create_train_dataset(self, train_df):
        if self.cfg.use_pseudo_mixup:
            
            print("Using BirdCLEFDatasetWithPseudoMixup for training...")
            return BirdCLEFDatasetWithPseudoMixup(
                df=train_df,
                cfg=self.cfg,
                spectrograms=self.spectrograms,
                pseudo_df=self.pseudo_df,
                pseudo_melspecs=self.pseudo_melspecs,
                mode="train",
                label2idx=self.label2index,
                idx2label=self.index2label
            )
        else:
            print("Using BirdCLEFDatasetFromNPY_Mixup for training...")
            return BirdCLEFDatasetFromNPY_Mixup(
                df=train_df,
                cfg=self.cfg,
                spectrograms=self.spectrograms,
                mode="train",
                label2idx=self.label2index,
                idx2label=self.index2label
            )

    def _calculate_auc(self, targets, outputs):
        probs = 1 / (1 + np.exp(-outputs))

        # 👇 ROC AUC はバイナリラベルを必要とするので、soft labelを2値化
        targets_bin = (targets >= 0.5).astype(int)

        aucs = [roc_auc_score(targets_bin[:, i], probs[:, i]) 
                for i in range(targets.shape[1]) if np.sum(targets_bin[:, i]) > 0]
        return np.mean(aucs) if aucs else 0.0

    def _calculate_classwise_auc(self, targets, outputs):
        probs = 1 / (1 + np.exp(-outputs))

        # バイナリ化（連続値でもintでも安全）
        targets_bin = (targets >= 0.5).astype(int)

        classwise_auc = {}
        for i in range(targets.shape[1]):
            if np.sum(targets_bin[:, i]) > 0:
                try:
                    classwise_auc[i] = roc_auc_score(targets_bin[:, i], probs[:, i])
                except ValueError:
                    classwise_auc[i] = np.nan  # エラー出たときも安心
        return classwise_auc

    def _calculate_classwise_ap(self, targets, outputs):
        probs = 1 / (1 + np.exp(-outputs))

        # ラベルをバイナリ化（soft label対応）
        targets_bin = (targets >= 0.5).astype(int)

        classwise_ap = {}
        for i in range(targets.shape[1]):
            if np.sum(targets_bin[:, i]) > 0:
                try:
                    classwise_ap[i] = average_precision_score(targets_bin[:, i], probs[:, i])
                except ValueError:
                    classwise_ap[i] = np.nan
        return classwise_ap
    
    def _calculate_map(self, targets, outputs):
        classwise_ap = self._calculate_classwise_ap(targets, outputs)
        values = [v for v in classwise_ap.values() if v is not None and not np.isnan(v)]
        return np.mean(values) if values else 0.0

    def _save_classwise_scores_to_csv(self, classwise_auc, classwise_ap, fold, filename_prefix):
        rows = []
        for i in classwise_auc:
            label = self.index2label.get(i, str(i))
            auc = classwise_auc[i]
            ap = classwise_ap.get(i, np.nan)
            rows.append({"label": label, "val_auc": auc, "val_ap": ap})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.cfg.model_path, f"{filename_prefix}_classwise_score_fold{fold}.csv"), index=False)


    def train_one_epoch(self, model, loader, optimizer, criterion, device, scheduler=None):
        model.train()
        losses, all_targets, all_outputs = [], [], []

        pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
        for step, batch in pbar:
            if isinstance(batch['melspec'], list):
                batch_outputs, batch_losses = [], []
                for i in range(len(batch['melspec'])):
                    inputs = batch['melspec'][i].unsqueeze(0).to(device)
                    target = batch['target'][i].unsqueeze(0).to(device)
                    optimizer.zero_grad()
            
                    output = model(inputs)
                    loss = criterion(output, target)
                    loss.backward()
                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())
                optimizer.step()
                outputs = torch.cat(batch_outputs, dim=0).numpy()
                loss = np.mean(batch_losses)
                targets = batch['target'].numpy()
            else:
                inputs = batch['melspec'].to(device)
                targets = batch['target'].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = outputs[1] if isinstance(outputs, tuple) else criterion(outputs, targets)
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                loss.backward()
                optimizer.step()
                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

            if scheduler and isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()

            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss.item() if not isinstance(loss, float) else loss)

            pbar.set_postfix({
                'train_loss': np.mean(losses[-10:]) if losses else 0,
                'lr': optimizer.param_groups[0]['lr']
            })

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        self.train_metrics = {
            'train_loss': np.mean(losses),
            'train_auc': self._calculate_auc(all_targets, all_outputs),
            "train_map": self._calculate_map(all_targets, all_outputs),   
            "train_classwise_auc": self._calculate_classwise_auc(all_targets, all_outputs),
            "train_classwise_ap": self._calculate_classwise_ap(all_targets, all_outputs),  
        }

    def validate(self, model, loader, criterion, device):
        model.eval()
        losses, all_targets, all_outputs = [], [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):
                if isinstance(batch['melspec'], list):
                    batch_outputs, batch_losses = [], []
                    for i in range(len(batch['melspec'])):
                        inputs = batch['melspec'][i].unsqueeze(0).to(device)
                        target = batch['target'][i].unsqueeze(0).to(device)
                        output = model(inputs)
                        loss = criterion(output, target)
                        batch_outputs.append(output.detach().cpu())
                        batch_losses.append(loss.item())
                    outputs = torch.cat(batch_outputs, dim=0).numpy()
                    loss = np.mean(batch_losses)
                    targets = batch['target'].numpy()
                else:
                    inputs = batch['melspec'].to(device)
                    targets = batch['target'].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    outputs = outputs.detach().cpu().numpy()
                    targets = targets.detach().cpu().numpy()

                all_outputs.append(outputs)
                all_targets.append(targets)
                losses.append(loss.item() if not isinstance(loss, float) else loss)

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        # print("Size of validation:",  len(all_targets))
        self.val_metrics = {
            'val_loss': np.mean(losses),
            'val_auc': self._calculate_auc(all_targets, all_outputs),
            "val_map": self._calculate_map(all_targets, all_outputs),
            "val_classwise_auc": self._calculate_classwise_auc(all_targets, all_outputs),
            "val_classwise_ap": self._calculate_classwise_ap(all_targets, all_outputs),
        }

    def run(self):
        
        for fold in range(self.cfg.n_fold):
            if fold not in self.cfg.selected_folds:
                continue
            print(f"\n{'='*30} Fold {fold} {'='*30}")

            # train.csvのfoldを使う．
            
            if self.cfg.full_train:
                train_df = self.df.reset_index(drop=True)
                val_df = self.df[self.df['fold'] == fold].reset_index(drop=True)
                print("Use full train data for training.")
            else:
                train_df = self.df[self.df['fold'] != fold].reset_index(drop=True)
                val_df = self.df[self.df['fold'] == fold].reset_index(drop=True) 
            
            print(f"Training set: {len(train_df)} samples")
            print(f"Validation set: {len(val_df)} samples")

            train_dataset = self._create_train_dataset(train_df)
            val_dataset = BirdCLEFDatasetFromNPY_Mixup(
                        df=val_df,
                        cfg=self.cfg,
                        spectrograms=self.spectrograms,
                        mode='valid',
                        label2idx=self.label2index,
                        idx2label=self.index2label
                    )

            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, 
                                       num_workers=self.cfg.num_workers, pin_memory=True,
                                       collate_fn=self.datasets_lib.collate_fn, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False,
                                     num_workers=self.cfg.num_workers, pin_memory=True,
                                     collate_fn=self.datasets_lib.collate_fn)
            # coatが文字列に含まれていれば
            if 'coat' in self.cfg.model_name:
                print("Using CoaT model")
                print(cfg.model_name)
                model = BirdCLEFModelForTrain_Coat(self.cfg).to(self.cfg.device)
            
            elif 'swin' in self.cfg.model_name:
                print("Using Swin model")
                print(cfg.model_name)
                model = BirdCLEFModelForTrain_Swin(self.cfg).to(self.cfg.device)
            else:
                print("efficientNet model")
                print(cfg.model_name)
                model = BirdCLEFModelForTrain(self.cfg).to(self.cfg.device)
                
                
                
            optimizer = self.learning_lib.get_optimizer(model, self.cfg)
            criterion = self.learning_lib.get_criterion(self.cfg)

            scheduler = (lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.lr, 
                        steps_per_epoch=len(train_loader), epochs=self.cfg.epochs, pct_start=0.1)
                         if self.cfg.scheduler == 'OneCycleLR'
                         else self.learning_lib.get_scheduler(optimizer, self.cfg))

            best_auc = 0
            log_history = []

            for epoch in range(self.cfg.epochs):
                print(f"\nEpoch {epoch+1}/{self.cfg.epochs}")
                start_time = time.time()

                self.train_one_epoch(model, train_loader, optimizer, criterion, self.cfg.device, scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None)
                self.validate(model, val_loader, criterion, self.cfg.device)

                # スコア取得
                train_loss = self.train_metrics['train_loss']
                train_auc = self.train_metrics['train_auc']
                train_auc_map = self.train_metrics['train_map']

                val_loss = self.val_metrics['val_loss']
                val_auc = self.val_metrics['val_auc']
                val_auc_map = self.val_metrics['val_map']
                val_classwise_auc = self.val_metrics['val_classwise_auc']
                val_classwise_ap = self.val_metrics['val_classwise_ap']

                if scheduler and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                    scheduler.step(val_loss if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau) else None)

                print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train MAP: {train_auc_map:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val MAP: {val_auc_map:.4f}")

                if val_auc > best_auc:
                    best_auc = val_auc
                    print(f"New best AUC: {best_auc:.4f} at epoch {epoch+1}")
                    
                    self._save_classwise_scores_to_csv(val_classwise_auc, val_classwise_ap, fold, filename_prefix="best_val")

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'epoch': epoch,
                        'val_auc': val_auc,
                        'train_auc': train_auc,
                        "index2label": self.index2label,
                        'cfg': self.cfg
                    }, f"{self.cfg.model_path}/model_fold{fold}.pth")

                log_entry = {
                    'epoch': epoch + 1,
                    'lr': scheduler.get_last_lr()[0] if scheduler else self.cfg.lr,
                    'epoch_time_min': round((time.time() - start_time) / 60, 2)
                }

                # classwiseスコアを除外した val_metrics のログ
                train_log = {f"{k}": v for k, v in self.train_metrics.items() if not k.startswith("train_classwise")}
                val_log = {f"{k}": v for k, v in self.val_metrics.items() if not k.startswith("val_classwise")}
                
                # ログ用スコアの更新（classwiseは除外）
                log_entry.update(train_log)
                log_entry.update(val_log)
                log_history.append(log_entry)
            

            pd.DataFrame(log_history).to_csv(f"{self.cfg.model_path}/log_fold{fold}.csv", index=False)
            self.best_scores.append(best_auc)
            print(f"\nBest AUC for fold {fold}: {best_auc:.4f}")

            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()

        print("\n" + "="*60)
        print("Cross-Validation Results:")
        for fold, score in enumerate(self.best_scores):
            print(f"Fold {self.cfg.selected_folds[fold]}: {score:.4f}")
        print(f"Mean AUC: {np.mean(self.best_scores):.4f}")
        print("="*60)


# In[8]:


# レア種はfold=-1にする．
def overwrite_fold_for_rare_classes(df, rare_threshold=5):
    # 各ラベルの出現数をカウント
    label_counts = df.groupby('primary_label').size()

    # rareなラベルをリストアップ
    rare_labels = label_counts[label_counts < rare_threshold].index.tolist()

    print(f"Rare labels ({len(rare_labels)} classes): {rare_labels[:10]}{'...' if len(rare_labels) > 10 else ''}")

    # rareなラベルのデータだけ fold = -1 に上書き
    df.loc[df['primary_label'].isin(rare_labels), 'fold'] = -1

    return df


# In[9]:


# モデルはmodels_{current_time}に保存される．
if __name__ == "__main__":
    utils_lib.set_seed(cfg.seed)
    print("\nLoading training data...")
    train_df = pd.read_csv(cfg.train_csv)
    
    if not cfg.secondary_labels:
        print("secondary_labels is not used.")
        train_df["secondary_labels"] = "['']"
    
    if cfg.is_RareFull: 
        print("Rare species are all in train fold.")
        train_df = overwrite_fold_for_rare_classes(train_df, rare_threshold=5)
        
    # taxonomyはラベルとindexの対応を取るために必要．
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    print("\nStarting training...")
    trainer = BirdCLEFTrainer(cfg, train_df, taxonomy_df,  datasets_lib, models_lib, learning_lib)
    trainer.run()
    print("\nTraining complete!")

