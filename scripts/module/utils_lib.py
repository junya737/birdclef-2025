import os

import numpy as np
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import torch

from IPython.display import Audio
import librosa


def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def create_submission(row_ids, predictions, species_ids, cfg):
    """Create submission dataframe"""
    print("Creating submission dataframe...")

    submission_dict = {'row_id': row_ids}
    
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]

    submission_df = pd.DataFrame(submission_dict)

    submission_df.set_index('row_id', inplace=True)

    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        for col in missing_cols:
            submission_df[col] = 0.0

    submission_df = submission_df[sample_sub.columns]

    submission_df = submission_df.reset_index()
    
    return submission_df

def play_audio(filename, base_path):
    """
    音声ファイルを再生する。
    
    Parameters:
    - filename: メタデータに含まれるファイルパス（例: '1139490/CSA36385.ogg'）
    - base_path: self.train_datadir に相当するルートパス
    """
    filepath = os.path.join(base_path, filename)
    if os.path.exists(filepath):
        return Audio(filename=filepath)
    else:
        print("ファイルが見つかりません:", filepath)
        

def inverse_melspec(mel_spec_norm, config):
    """
    正規化された Mel スペクトログラム（0〜1）を元の波形に近づけて再生。
    
    Parameters:
    - mel_spec_norm: 正規化済みメルスペクトログラム（float32, shape: (128, T)）
    - config: 設定オブジェクト（DatasetConfig）

    Returns:
    - IPython.display.Audio オブジェクト
    """

    # ① 0〜1 の正規化を元に戻す
    mel_spec_db = mel_spec_norm * (0 - (-80)) + (-80)  # librosa.power_to_db の typical range: [-80, 0]

    # ② dB → power
    mel_power = librosa.db_to_power(mel_spec_db)

    # ③ mel → STFT
    stft = librosa.feature.inverse.mel_to_stft(
    M=mel_power,
    sr=config.FS,
    n_fft=config.N_FFT,
    power=2.0,
    fmin=config.FMIN,
    fmax=config.FMAX
    )

    # ④ Griffin-Lim で波形復元
    waveform = librosa.griffinlim(
        stft,
        n_iter=32,
        hop_length=config.HOP_LENGTH,
        win_length=config.N_FFT
    )

    return Audio(waveform, rate=config.FS)
