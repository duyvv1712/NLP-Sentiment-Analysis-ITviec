"""
Pipeline Bước 1 - Tiền xử lý dữ liệu
Sử dụng pyvi thay underthesea để tránh lỗi torch/WinError 1455
"""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from preprocessing import TextPreprocessor

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE, '..')
BATCH_SIZE = 500

# ─── 1. Đọc dữ liệu thô ───────────────────────────────────────────────────────
print("\n[1/7] Đọc dữ liệu thô Reviews.xlsx ...")
raw_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Reviews.xlsx')
df = pd.read_excel(raw_path)
print(f"    ✓ Tổng số mẫu: {len(df):,} | Số cột: {df.shape[1]}")

# ─── 2. Ghép văn bản ──────────────────────────────────────────────────────────
print("\n[2/7] Ghép nội dung đánh giá ...")
title   = df.get('Title', pd.Series([''] * len(df))).fillna('').astype(str)
liked   = df.get('What I liked', pd.Series([''] * len(df))).fillna('').astype(str)
suggest = df.get('Suggestions for improvement', pd.Series([''] * len(df))).fillna('').astype(str)
df['raw_review_text'] = title + ' . ' + liked + ' . ' + suggest
print(f"    ✓ Ví dụ: {df['raw_review_text'].iloc[0][:100]} ...")

# ─── 3. Khởi tạo TextPreprocessor ────────────────────────────────────────────
print("\n[3/7] Khởi tạo module TextPreprocessor ...")
dict_dir = os.path.join(PROJECT_ROOT, 'data', 'dictionaries')
tp = TextPreprocessor(dict_dir=dict_dir)
print(f"    ✓ Stopwords: {len(tp.stopwords):,} | Teencode: {len(tp.teencode_dict):,} | ENG-VN: {len(tp.english_vnmese_dict):,}")

# Test tokenizer ngay từ đầu để xác nhận hoạt động
print("    Kiểm tra tokenizer ...")
test_result = tp.clean_advance_text("Môi trường tốt và thoải mái", remove_stopwords=True)
print(f"    ✓ Test tokenize: '{test_result}'")

# ─── 4. Clean Basic Text ──────────────────────────────────────────────────────
print("\n[4/7] Áp dụng clean_basic_text ...")
df['clean_basic_text'] = df['raw_review_text'].progress_apply(tp.clean_basic_text)
print(f"    ✓ Ví dụ: {df['clean_basic_text'].iloc[0][:100]} ...")

# ─── 5. Trích xuất đặc trưng Lexicon ─────────────────────────────────────────
print("\n[5/7] Trích xuất đặc trưng Lexicon ...")
lex_feats = df.progress_apply(
    lambda row: tp.calc_sentiment_features(row['clean_basic_text'], raw_text=row['raw_review_text']),
    axis=1
)
lex_df = pd.DataFrame(list(lex_feats))
for col in lex_df.columns:
    df[col] = lex_df[col].values
lexicon_coverage = (df['total_we'] > 0).sum()
print(f"    ✓ Reviews có tín hiệu lexicon: {lexicon_coverage:,}/{len(df):,} ({lexicon_coverage/len(df)*100:.1f}%)")

# ─── 6. Clean Advance Text - BATCH với gc ─────────────────────────────────────
print(f"\n[6/7] Tách từ tiếng Việt (batch {BATCH_SIZE} mẫu) ...")
texts = df['raw_review_text'].tolist()
total = len(texts)
advance_results = []

num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
for b in range(num_batches):
    start = b * BATCH_SIZE
    end   = min(start + BATCH_SIZE, total)
    batch = texts[start:end]
    print(f"    Batch {b+1}/{num_batches} ({start}–{end}) ...", flush=True)

    batch_results = []
    for text in batch:
        r = tp.clean_advance_text(text, remove_stopwords=True)
        batch_results.append(r)
    advance_results.extend(batch_results)

    del batch_results
    gc.collect()
    print(f"    ✓ Batch {b+1} xong: {end}/{total} mẫu", flush=True)

df['clean_advance_text'] = advance_results[:len(df)]
print(f"\n    ✓ Tách từ hoàn thành! Ví dụ[0]: {df['clean_advance_text'].iloc[0][:100]} ...")

# ─── 7. Gán nhãn cảm xúc ──────────────────────────────────────────────────────
print("\n[7/7] Gán nhãn cảm xúc từ Rating ...")
df['sentiment'] = df['Rating'].apply(tp.map_sentiment_label)

dist = df['sentiment'].value_counts()
print(f"    ✓ Phân bố nhãn:")
for label, count in dist.items():
    pct = count / len(df) * 100
    print(f"       {label:10s}: {count:,} mẫu ({pct:.1f}%)")

# ─── 8. Xuất kết quả ─────────────────────────────────────────────────────────
out_dir  = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(out_dir, exist_ok=True)
out_xlsx = os.path.join(out_dir, 'reviews_cleaned.xlsx')
out_csv  = os.path.join(out_dir, 'reviews_cleaned.csv')

df.to_excel(out_xlsx, index=False)
df.to_csv(out_csv, index=False, encoding='utf-8-sig')

print(f"\n{'='*62}")
print(f"OK HOAN THANH BUOC 1 - TIEN XU LY DU LIEU")
print(f"{'='*62}")
print(f"   File Excel : {out_xlsx}")
print(f"   File CSV   : {out_csv}")
print(f"   Tong mau   : {len(df):,} hang | {df.shape[1]} cot")
print(f"   Cac cot moi: raw_review_text, clean_basic_text, clean_advance_text,")
print(f"                pos_w, neg_w, pos_e, neg_e, total_we, sentiment_ratio, sentiment")
print(f"{'='*62}")
