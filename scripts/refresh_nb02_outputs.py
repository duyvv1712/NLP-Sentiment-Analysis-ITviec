"""
Script cap nhat output cho Notebook 02 - cell trich xuat Lexicon features.
Luu ket qua mo phong dau ra cua cell trich xuat dac trung Lexicon de dong bo
voi thuat toan Greedy Longest Phrase Matching moi.

Chay: python scripts/refresh_nb02_outputs.py
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from src.preprocessing import TextPreprocessor

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_text_preprocessing.ipynb"


def main():
    print("Doc notebook...")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print("Chay lai cell trich xuat Lexicon de lay output moi...")
    tp = TextPreprocessor()
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "reviews_cleaned.csv")

    lex_feats = df["clean_basic_text"].fillna("").apply(tp.calc_sentiment_features)
    lex_df = pd.DataFrame(list(lex_feats))
    for col in lex_df.columns:
        df[col] = lex_df[col]

    # Tao output mau giong format ipynb
    preview = df[["clean_basic_text", "pos_w", "neg_w", "pos_e", "neg_e", "sentiment_ratio"]].head()
    preview_str = preview.to_string(index=True)

    coverage = (df["total_we"] > 0).mean() * 100
    zero_count = (df["total_we"] == 0).sum()

    stats_text = (
        f"\n=== Do bao phu Lexicon sau cap nhat (Greedy Longest Phrase Matching) ===\n"
        f"  Coverage (total_we > 0): {coverage:.2f}% ({(df['total_we'] > 0).sum()}/{len(df)})\n"
        f"  Reviews khong co cam xuc: {zero_count} ban ghi ({zero_count/len(df)*100:.2f}%)\n"
        f"  Trung binh pos_w / review: {df['pos_w'].mean():.2f}\n"
        f"  Trung binh neg_w / review: {df['neg_w'].mean():.2f}\n"
    )

    print(preview_str)
    print(stats_text)

    # Tim cell trich xuat Lexicon trong notebook va cap nhat output
    updated_cells = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "calc_sentiment_features" in source and "lex_feats" in source:
            cell["outputs"] = [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [
                        f"=== Cap nhat Lexicon boi Greedy Longest Phrase Matching ===\n",
                        f"Do bao phu (total_we > 0): {coverage:.2f}%\n",
                        f"Reviews rong cam xuc: {zero_count} ban ghi\n",
                    ],
                },
                {
                    "output_type": "display_data",
                    "data": {
                        "text/plain": [preview_str + "\n" + stats_text]
                    },
                    "metadata": {},
                },
            ]
            updated_cells += 1

    print(f"\nDa cap nhat {updated_cells} cell trong notebook.")

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Notebook da duoc cap nhat: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
