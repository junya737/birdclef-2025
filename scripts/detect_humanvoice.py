import os
import pandas as pd
import torch
from glob import glob
from tqdm import tqdm
from datetime import datetime, timedelta, timezone


def detect_speech_segments(audio_dir, debug=False, debug_count=50, threshold=0.5):
    print("🔍 Silero VAD モデルの読み込み中...")
    model, (get_speech_timestamps, _, read_audio, _, _) = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )

    print(f"📂 音声ファイルの読み込み元: {audio_dir}")
    ogg_files = sorted(glob(os.path.join(audio_dir, "*", "*.ogg")))
    print(f"🎧 全ファイル数: {len(ogg_files)}")

    if debug:
        ogg_files = ogg_files[:debug_count]
        print(f"🚨 [DEBUG MODE] 最初の {debug_count} ファイルのみ処理します")

    speech_rows = []

    for fname in tqdm(ogg_files):
        try:
            wav = read_audio(fname)
            speech_segments = get_speech_timestamps(wav, model, return_seconds=True, threshold=threshold)

            for seg in speech_segments:
                speech_rows.append({
                    "filename": os.path.relpath(fname, audio_dir),
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2)
                })

        except Exception as e:
            print(f"[Error] {fname}: {e}")

    speech_df = pd.DataFrame(speech_rows)

    # === JST時刻で保存ディレクトリ作成 ===
    JST = timezone(timedelta(hours=9), 'JST')
    timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M")
    output_dir = f"../data/processed/human_voice_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "human_voice.csv")
    speech_df.to_csv(output_csv, index=False)

    print(f"\n✅ 検出結果を保存しました: {output_csv}")
    print(speech_df.head())

# ==== 実行例 ====
AUDIO_DIR = "../data/raw/train_audio"


detect_speech_segments(AUDIO_DIR, debug=False)

