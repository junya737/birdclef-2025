import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import os

import gc
import time

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json




def get_optimizer(model, cfg):
  
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")
        
    return optimizer

def get_scheduler(optimizer, cfg):
   
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == 'OneCycleLR':
        scheduler = None  
    else:
        scheduler = None
        
    return scheduler

def get_criterion(cfg):
 
    if cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")
        
    return criterion


class BirdCLEFTrainer:
    def __init__(self, cfg, df, datasets_lib, models_lib, learning_lib):
        self.cfg = cfg
        self.df = df.head(100).reset_index(drop=True) if cfg.debug else df
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

        self._setup_model_dir()
        self._save_config()
        self._load_taxonomy()
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

    def _load_taxonomy(self):
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        species_ids = taxonomy_df['primary_label'].tolist()
        self.cfg.num_classes = len(species_ids)
        # labelとindexの対応
        self.index2label = {i: label for i, label in enumerate(species_ids)}
        self.label2index = {label: i for i, label in enumerate(species_ids)}

    def _load_spectrograms(self):
        print(f"Loading pre-computed mel spectrograms from NPY file, from the path: {self.cfg.spectrogram_npy}")
        self.spectrograms = np.load(self.cfg.spectrogram_npy, allow_pickle=True).item()
        print(f"Loaded {len(self.spectrograms)} pre-computed mel spectrograms")
        
    def _load_pseudo_data(self):
        print("📥 Loading pseudo label CSV and melspecs...")

        # row_id を index にして読み込む（← ここがポイント！）
        self.pseudo_df = pd.read_csv(self.cfg.pseudo_label_csv, index_col="row_id")

        # 信頼度フィルタリング（例: 最大値が 0.5 未満の行を除く）
        confidence_threshold = self.cfg.pseudo_conf_threshold
        max_probs = self.pseudo_df.max(axis=1)
        self.pseudo_df = self.pseudo_df[max_probs > confidence_threshold]
        self.pseudo_df = self.pseudo_df.reset_index(drop=False)
        print(f"✅ Filtered pseudo labels: {len(self.pseudo_df)}")

        # melspec は key が row_id の dict を想定
        self.pseudo_melspecs = np.load(self.cfg.pseudo_melspec_npy, allow_pickle=True)
        print(f"✅ Loaded pseudo mel-spectrograms: {len(self.pseudo_melspecs)}")
        
    def _create_train_dataset(self, train_df):
        if self.cfg.use_pseudo_mixup:
            print("🔀 Using BirdCLEFDatasetWithPseudo (with pseudo label mixup)")
            return self.datasets_lib.BirdCLEFDatasetWithPseudo(
                train_df=train_df,
                pseudo_df=self.pseudo_df,
                cfg=self.cfg,
                spectrograms=self.spectrograms,
                pseudo_melspecs=self.pseudo_melspecs,
                mode='train'
            )
        else:
            print("📦 Using BirdCLEFDatasetFromNPY (no pseudo mixup)")
            return self.datasets_lib.BirdCLEFDatasetFromNPY(
                train_df, self.cfg, self.spectrograms, mode='train'
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
        self.val_metrics = {
            'val_loss': np.mean(losses),
            'val_auc': self._calculate_auc(all_targets, all_outputs),
            "val_map": self._calculate_map(all_targets, all_outputs),
            "val_classwise_auc": self._calculate_classwise_auc(all_targets, all_outputs),
            "val_classwise_ap": self._calculate_classwise_ap(all_targets, all_outputs),
        }

    def run(self):
        skf = StratifiedKFold(n_splits=self.cfg.n_fold, shuffle=True, random_state=self.cfg.seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.df, self.df['primary_label'])):
            if fold not in self.cfg.selected_folds:
                continue
            print(f"\n{'='*30} Fold {fold} {'='*30}")

            train_df = self.df.iloc[train_idx].reset_index(drop=True)
            val_df = self.df.iloc[val_idx].reset_index(drop=True)
            print(f"Training set: {len(train_df)} samples")
            print(f"Validation set: {len(val_df)} samples")

            train_dataset = self._create_train_dataset(train_df)
            val_dataset = self.datasets_lib.BirdCLEFDatasetFromNPY(
                val_df, self.cfg, self.spectrograms, mode='valid'
            )

            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, 
                                       num_workers=self.cfg.num_workers, pin_memory=True,
                                       collate_fn=self.datasets_lib.collate_fn, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False,
                                     num_workers=self.cfg.num_workers, pin_memory=True,
                                     collate_fn=self.datasets_lib.collate_fn)

            model = self.models_lib.BirdCLEFModelForTrain(self.cfg).to(self.cfg.device)
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

