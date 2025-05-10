import os

import numpy as np
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import torch

from IPython.display import Audio
import librosa
import matplotlib.pyplot as plt


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


def plot_melspectrogram(spec: dict, species_id: str, 
                        fmin: int = 50, fmax: int = 16000, 
                        duration: float = 5.0,
                        figsize=(6, 4), cmap='magma'):

    if species_id not in spec:
        print(f"[ERROR] '{species_id}' not found in spec!")
        return

    mel = spec[species_id]  # shape: (n_mels, time_steps)
    extent = [0, duration, fmin, fmax]

    plt.figure(figsize=figsize)
    plt.imshow(mel, aspect='auto', origin='lower', cmap=cmap, extent=extent)
    plt.title(f"MelSpectrogram of {species_id}")
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.show()
    
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio

def plot_and_play_audio(filename, base_dir, sr=32000):
    """
    音声ファイルをロードして、波形プロットと再生を同時に行う
    """
    filepath = os.path.join(base_dir, filename)
    y, _ = librosa.load(filepath, sr=sr)

    # 波形プロット
    plt.figure(figsize=(14, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform: {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    # 音声再生
    return Audio(y, rate=sr)


import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from IPython.display import Audio

def seconds_and_minutes_formatter(x, pos):
    """秒数と分:秒表記を同時に表示するフォーマッタ"""
    minutes = int(x) // 60
    seconds = int(x) % 60
    return f"{int(x)}s\n({minutes}:{seconds:02d})"

#文字の大きさ

def plot_power(rec, base_path='../data/raw/train_audio/', chunk_len=0.05):
    """
    指定された音声ファイルを読み込み、パワー（dB）を時間方向にプロットする関数
    """
    filepath = f'{base_path}/{rec}'

    # 音声ロード
    wav, sr = librosa.load(filepath)
    
    # ★ここで長さを表示！
    print(f"Audio length: {len(wav)/sr:.2f} seconds")

    # パワー計算
    power = wav ** 2

    # チャンクごとにパワー合計
    chunk = int(chunk_len * sr)
    pad = int(np.ceil(len(power) / chunk) * chunk - len(power))
    power = np.pad(power, (0, pad))
    power = power.reshape((-1, chunk)).sum(axis=1)

    # 時間軸
    t = np.arange(len(power)) * chunk_len

    # プロット
    fig, ax = plt.subplots(figsize=(24, 6))
    ax.plot(t, 10 * np.log10(power + 1e-12))  # log(0)対策
    ax.set_title(f"Recording: {rec}")
    ax.set_xlabel("Time [sec] (min:sec)")
    ax.set_ylabel("Power (dB)")

    # 横軸フォーマット変更
    ax.xaxis.set_major_formatter(FuncFormatter(seconds_and_minutes_formatter))

    # 10秒ごとに目盛りを設定
    ax.xaxis.set_major_locator(MultipleLocator(10))

    ax.grid(True)

    plt.show()

    return Audio(filepath, rate=sr)