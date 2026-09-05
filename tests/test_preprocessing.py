import pytest
import pandas as pd
from pathlib import Path
from src.preprocessing import TextPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DICT_DIR = PROJECT_ROOT / "data" / "dictionaries"
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "reviews_cleaned.csv"


@pytest.fixture(scope="module")
def preprocessor():
    return TextPreprocessor(dict_dir=str(DICT_DIR))


def test_che_do_bao_hiem_tot(preprocessor):
    """Kiểm tra câu 'chế độ bảo hiểm tốt' được match chính xác thành thuộc tính tích cực."""
    text = "chế độ bảo hiểm tốt"
    res = preprocessor.calc_sentiment_features(text)
    assert res["pos_w"] >= 1
    assert res["neg_w"] == 0
    assert res["sentiment_ratio"] > 0


def test_common_positive_phrases(preprocessor):
    """Kiểm tra các cụm từ tích cực đa âm tiết phổ biến."""
    text = "môi trường làm việc thoải mái vui vẻ và thân thiện"
    res = preprocessor.calc_sentiment_features(text)
    assert res["pos_w"] >= 3
    assert res["neg_w"] == 0
    assert res["sentiment_ratio"] == 1.0


def test_negation_handling(preprocessor):
    """Kiểm tra thuật ngữ phủ định: 'không toxic' -> POS, 'không tăng lương' -> NEG."""
    pos_negation = "công ty không toxic văn hóa tốt"
    res_pos = preprocessor.calc_sentiment_features(pos_negation)
    assert res_pos["pos_w"] >= 2
    assert res_pos["neg_w"] == 0
    assert res_pos["sentiment_ratio"] == 1.0

    neg_phrase = "ở đây không tăng lương sếp khó tính"
    res_neg = preprocessor.calc_sentiment_features(neg_phrase)
    assert res_neg["pos_w"] == 0
    assert res_neg["neg_w"] >= 2
    assert res_neg["sentiment_ratio"] == -1.0


def test_word_tokenize_with_underscores(preprocessor):
    """Kiểm tra tương thích với văn bản có dấu gạch dưới từ underthesea word_tokenize."""
    text_with_underscores = "môi_trường thoải_mái chế_độ bảo_hiểm tốt công_ty năng_động"
    res = preprocessor.calc_sentiment_features(text_with_underscores)
    assert res["pos_w"] >= 3
    assert res["neg_w"] == 0


def test_empty_and_invalid_inputs(preprocessor):
    """Kiểm tra xử lý văn bản rỗng hoặc không hợp lệ."""
    for invalid in ["", "   ", None, 12345]:
        res = preprocessor.calc_sentiment_features(invalid)
        assert res["pos_w"] == 0
        assert res["neg_w"] == 0
        assert res["total_we"] == 0
        assert res["sentiment_ratio"] == 0.0


def test_emoji_detection(preprocessor):
    """Kiểm tra nhận diện emoji cảm xúc."""
    text_emoji = "công ty tốt lắm 😍 ❤️"
    res = preprocessor.calc_sentiment_features(text_emoji)
    assert res["pos_w"] >= 1
    assert res["pos_e"] >= 2


def test_dataset_high_coverage(preprocessor):
    """Kiểm tra độ bao phủ trên mẫu 200 đánh giá thực tế từ reviews_cleaned.csv đạt trên 95%."""
    if not DATA_FILE.exists():
        pytest.skip("Tệp reviews_cleaned.csv không tồn tại.")
    df = pd.read_csv(DATA_FILE, nrows=200)
    texts = df["clean_basic_text"].fillna("")
    matched_count = 0
    for text in texts:
        res = preprocessor.calc_sentiment_features(text)
        if res["total_we"] > 0:
            matched_count += 1
    coverage = matched_count / len(df)
    assert coverage >= 0.95, f"Độ bao phủ thực tế chỉ đạt {coverage * 100:.2f}%, yêu cầu >= 95%"
