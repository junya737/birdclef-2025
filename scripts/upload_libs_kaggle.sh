#!/bin/bash

DATASET_DIR="."

# ディレクトリ確認
if [ ! -d "$DATASET_DIR/module" ]; then
  echo "エラー: $DATASET_DIR/module が見つかりません。ディレクトリ構成を確認してください。"
  exit 1
fi



# アップロード実行
if kaggle datasets status -p "$DATASET_DIR" 2>&1 | grep -q "not found"; then
  echo "初回アップロード中..."
  kaggle datasets create -p "$DATASET_DIR" --dir-mode zip
else
  echo "更新中..."
  kaggle datasets version -p "$DATASET_DIR" --dir-mode zip -m "Updated at $(date)"
fi