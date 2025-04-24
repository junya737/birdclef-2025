import os
import pandas as pd
import librosa
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, freeze_support
from datetime import datetime, timedelta, timezone

# ===== 無音検出関数（パワー閾値ベース） =====
def detect_silence_segments(filepath, audio_dir, chunk_len_sec=0.1, power_thresh_db=-40.0):
    try:
        y, sr = librosa.load(filepath, sr=32000)
        power = y ** 2
        chunk_size = int(chunk_len_sec * sr)

        pad_len = (chunk_size - len(power) % chunk_size) % chunk_size
        power = np.pad(power, (0, pad_len), mode='constant')
        power_chunks = power.reshape(-1, chunk_size).mean(axis=1)
        power_db = 10 * np.log10(power_chunks + 1e-10)

        silence_mask = power_db < power_thresh_db

        silence_segments = []
        is_silent = False
        start = 0

        for i, silent in enumerate(silence_mask):
            if silent and not is_silent:
                start = i * chunk_len_sec
                is_silent = True
            elif not silent and is_silent:
                end = i * chunk_len_sec
                silence_segments.append({
                    "filename": os.path.relpath(filepath, audio_dir),
                    "start": round(start, 2),
                    "end": round(end, 2)
                })
                is_silent = False

        if is_silent:
            end = len(silence_mask) * chunk_len_sec
            silence_segments.append({
                "filename": os.path.relpath(filepath, audio_dir),
                "start": round(start, 2),
                "end": round(end, 2)
            })

        return silence_segments

    except Exception as e:
        print(f"[Error] {filepath}: {e}")
        return []


# ===== 並列で全ファイルに対して実行 =====
def run_silence_detection_all(audio_dir, n_jobs=4, debug=False, debug_count=50):
    ogg_files = sorted(glob(os.path.join(audio_dir, "*", "*.ogg")))
    print(f"🎧 対象ファイル数: {len(ogg_files)}")

    if debug:
        ogg_files = ogg_files[:debug_count]
        print(f"🚨 [DEBUG MODE] 最初の {debug_count} ファイルのみ処理します")

    tasks = [(fname, audio_dir) for fname in ogg_files]

    with Pool(n_jobs) as pool:
        results = list(tqdm(pool.starmap(detect_silence_segments, tasks), total=len(tasks)))

    # 平坦化
    flat_results = [item for sublist in results for item in sublist]
    silence_df = pd.DataFrame(flat_results)

    # JSTで保存ディレクトリを作成
    JST = timezone(timedelta(hours=9), 'JST')
    timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M")
    output_dir = f"../data/processed/silence_segments_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "silence_segments.csv")
    silence_df.to_csv(output_csv, index=False)
    print(f"\n✅ 無音区間を保存しました: {output_csv}")
    print(silence_df.head())


# ===== 実行例 =====
if __name__ == "__main__":
    freeze_support()  # Windows 対策
    AUDIO_DIR = "../data/raw/train_audio"
    run_silence_detection_all(AUDIO_DIR, n_jobs=32, debug=False)