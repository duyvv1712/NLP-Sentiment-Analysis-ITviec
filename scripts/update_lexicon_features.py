"""
Script cập nhật lại 6 cột Lexicon (pos_w, neg_w, pos_e, neg_e, total_we, sentiment_ratio)
trong reviews_cleaned.csv và reviews_cleaned.xlsx bằng thuật toán Greedy Longest Phrase Matching mới.
Chạy từ thư mục gốc của dự án: python scripts/update_lexicon_features.py
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from src.preprocessing import TextPreprocessor

CSV_PATH  = PROJECT_ROOT / "data" / "processed" / "reviews_cleaned.csv"
XLSX_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_cleaned.xlsx"

LEXICON_COLS = ["pos_w", "neg_w", "pos_e", "neg_e", "total_we", "sentiment_ratio"]

def main():
    print("Đang tải dữ liệu...")
    df = pd.read_csv(CSV_PATH)
    tp = TextPreprocessor()

    print(f"Tổng số bản ghi: {len(df)}")
    print("Đang tính lại đặc trưng Lexicon bằng thuật toán Greedy Longest Phrase Matching...")

    lex_feats = df["clean_basic_text"].fillna("").apply(tp.calc_sentiment_features)
    lex_df = pd.DataFrame(list(lex_feats))

    for col in LEXICON_COLS:
        df[col] = lex_df[col]

    coverage = (df["total_we"] > 0).mean() * 100
    pos_cover = (df["pos_w"] > 0).mean() * 100
    neg_cover = (df["neg_w"] > 0).mean() * 100
    zero_count = (df["total_we"] == 0).sum()

    print("\n=== KẾT QUẢ SAU CẬP NHẬT ===")
    print(f"  Độ bao phủ (total_we > 0): {coverage:.2f}%")
    print(f"  Đánh giá có pos_w > 0    : {pos_cover:.2f}%")
    print(f"  Đánh giá có neg_w > 0    : {neg_cover:.2f}%")
    print(f"  Đánh giá rỗng cảm xúc   : {zero_count} bản ghi ({(df['total_we'] == 0).mean()*100:.2f}%)")
    print(f"  Trung bình pos_w / review: {df['pos_w'].mean():.2f}")
    print(f"  Trung bình neg_w / review: {df['neg_w'].mean():.2f}")

    print("\nĐang lưu lại reviews_cleaned.csv...")
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print("Đang lưu lại reviews_cleaned.xlsx...")
    df.to_excel(XLSX_PATH, index=False)

    print("\nHoàn thành! Cả hai file đã được cập nhật đồng bộ.")

if __name__ == "__main__":
    main()
