import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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


class BirdCLEFDatasetFromNPY_Labeled(Dataset):
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
        
        # filename: 126247/XC941297.ogg samplename: 126247-XC941297
        # spectrograms のkey がsamplename みたいになっているから．
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        sample_names = set(self.df['samplename'])
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples")
        
        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
            
        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":  # Only print warning during training
                print(f"Warning: Spectrogram for {samplename} not found and could not be generated")

        "TODO: 3chに対応できるように．"
        
        spec = torch.tensor(spec, dtype=torch.float32)
        if spec.ndim == 2:
            # (H, W) → (1, H, W)
            spec = spec.unsqueeze(0)
        elif spec.ndim == 3 and spec.shape[0] != 1:
            # already (3, H, W) etc → do nothing
            pass
        else:
            raise ValueError(f"Unexpected spectrogram shape: {spec.shape}")


        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)
        
        target = self.encode_label(row['primary_label'])
        
        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']
            
            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0
        
        return {
            'melspec': spec, 
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }
    
    def apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram"""
    
        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0
        
        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0
        
        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1) 
            
        return spec
    
    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target
    



class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode

        self.spectrograms = spectrograms
        
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename
        
        # filename: 126247/XC941297.ogg samplename: 126247-XC941297
        # spectrograms のkey がsamplename みたいになっているから．
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        sample_names = set(self.df['samplename'])
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples")
        
        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
            
        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":  # Only print warning during training
                print(f"Warning: Spectrogram for {samplename} not found and could not be generated")

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)
        
        target = self.encode_label(row['primary_label'])
        
        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']
            
            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0
        
        return {
            'melspec': spec, 
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }
    
    def apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram"""
    
        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0
        
        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0
        
        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1) 
            
        return spec
    
    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target
    

def collate_fn(batch):
    """Custom collate function to handle different sized spectrograms"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
        
    result = {key: [] for key in batch[0].keys()}
    
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    
    for key in result:
        if key == 'target' and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == 'melspec' and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(result[key])
    
    return result





class BirdCLEFDatasetFromNPY_CELoss(Dataset):
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
        
        # filename: 126247/XC941297.ogg samplename: 126247-XC941297
        # spectrograms のkey がsamplename みたいになっているから．
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        sample_names = set(self.df['samplename'])
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples")
        
        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
            
        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":  # Only print warning during training
                print(f"Warning: Spectrogram for {samplename} not found and could not be generated")

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)
        
        target = self.encode_label(row['primary_label'])
        
        return {
            'melspec': spec, 
            'target': torch.tensor(target, dtype=torch.long),
            'filename': row['filename']
        }
    
    def apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram"""
    
        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0
        
        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0
        
        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1) 
            
        return spec
    
    def encode_label(self, label):
        """Encode label to integer index"""
        if label in self.label_to_idx:
            return self.label_to_idx[label]
        else:
            return -1  # エラー防止
    


# pseudo labelsをmixupするデータセット．
class BirdCLEFDatasetWithPseudo(Dataset):
    def __init__(self, train_df, pseudo_df, cfg, spectrograms, pseudo_melspecs, mode="train"):
        self.train_df = train_df.reset_index(drop=True)
        self.pseudo_df = pseudo_df.set_index("row_id")
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms
        self.pseudo_melspecs = pseudo_melspecs
        self.use_pseudo_mixup = cfg.use_pseudo_mixup
        self.pseudo_mix_prob = cfg.pseudo_mix_prob  # ← cfgから取得

        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        self.train_df["samplename"] = self.train_df["filename"].apply(
            lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0]
        )
    
    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        row = self.train_df.iloc[idx]
        samplename = row["samplename"]
        spec = self.spectrograms.get(samplename, np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32))
        label = self._encode_label(row)

        if self.mode == "train" and self.use_pseudo_mixup and random.random() < self.pseudo_mix_prob:
            pseudo_row_id = random.choice(self.pseudo_df.index)
            pseudo_spec = self.pseudo_melspecs.get(pseudo_row_id, np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32))
            pseudo_label = self.pseudo_df.loc[pseudo_row_id][self.species_ids].values.astype(np.float32)

            lam = np.random.beta(self.cfg.mixup_alpha_real, self.cfg.mixup_alpha_pseudo)
            spec = lam * spec + (1 - lam) * pseudo_spec
            label = np.maximum(label, pseudo_label)

        return {
            "melspec": torch.tensor(spec, dtype=torch.float32).unsqueeze(0),
            "target": torch.tensor(label, dtype=torch.float32),
            "filename": row["filename"]
        }
    
    def _encode_label(self, row):
        target = np.zeros(self.num_classes, dtype=np.float32)
        if row["primary_label"] in self.label_to_idx:
            target[self.label_to_idx[row["primary_label"]]] = 1.0

        if "secondary_labels" in row and pd.notnull(row["secondary_labels"]):
            try:
                if isinstance(row["secondary_labels"], str):
                    secondary = eval(row["secondary_labels"])
                else:
                    secondary = row["secondary_labels"]

                for label in secondary:
                    if label in self.label_to_idx:
                        target[self.label_to_idx[label]] = 1.0
            except Exception:
                pass

        return target
        