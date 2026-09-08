import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_services import (
    REVIEWS_PATH,
    company_summary,
    get_model_status,
    load_reviews,
    sentiment_summary,
    top_terms,
)



def sample_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Company Name": ["Alpha", "Alpha", "Alpha", "Beta"],
            "Rating": [5, 4, 1, 3],
            "sentiment": ["Positive", "Positive", "Negative", "Neutral"],
            "clean_advance_text": [
                "môi_trường tốt",
                "phúc_lợi tốt",
                "quản_lý tệ",
                "bình_thường",
            ],
        }
    )


def test_sentiment_summary_keeps_all_three_classes():
    summary = sentiment_summary(sample_reviews().iloc[:2])

    assert summary["sentiment"].tolist() == ["Positive", "Neutral", "Negative"]
    assert summary["reviews"].tolist() == [2, 0, 0]
    assert summary["share"].tolist() == [100.0, 0.0, 0.0]


def test_dashboard_uses_fast_csv_source_with_normalized_newlines():
    reviews = load_reviews()

    assert REVIEWS_PATH.suffix == ".csv"
    assert len(reviews) == 8_417
    assert set(reviews.columns).issuperset(
        {"Company Name", "Rating", "clean_advance_text", "sentiment"}
    )
    assert not reviews["What I liked"].dropna().str.contains("\r\n", regex=False).any()


def test_company_summary_applies_sample_threshold_and_percentages():
    result = company_summary(sample_reviews(), min_reviews=2)

    assert result["Company Name"].tolist() == ["Alpha"]
    assert result.loc[0, "reviews"] == 3
    assert result.loc[0, "positive_share"] == pytest.approx(200 / 3)
    assert result.loc[0, "negative_share"] == pytest.approx(100 / 3)


def test_top_terms_counts_preprocessed_tokens():
    terms = top_terms(["môi_trường tốt", "phúc_lợi tốt"], limit=2)

    assert terms.to_dict("records") == [
        {"term": "tốt", "count": 2},
        {"term": "môi_trường", "count": 1},
    ]


def test_model_status_fails_closed_when_model_is_missing(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "text_feature_extractor.joblib").write_bytes(b"placeholder")
    (models / "artifact_manifest.json").write_text(
        json.dumps({"feature_contract": {"feature_mode": "text_only"}}),
        encoding="utf-8",
    )

    status = get_model_status(tmp_path)

    assert not status.ready
    assert status.model_path is None
    assert "TV3" in status.message


def test_dashboard_services_do_not_eagerly_import_nlp_stack():
    script = """
import sys
import src.app_services

assert "src.features" not in sys.modules
assert "src.preprocessing" not in sys.modules
assert "underthesea" not in sys.modules

import src.preprocessing
assert "underthesea" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_predict_review_hybrid_on_complex_negation():
    """Kiểm tra câu phủ định ghép nhiều vế được nhận diện chính xác là Negative qua cơ chế Hybrid."""
    from src.app_services import get_model_status, load_inference_bundle, predict_review
    from src.preprocessing import TextPreprocessor

    status = get_model_status()
    if not status.ready:
        pytest.skip("Model chưa sẵn sàng để test.")

    model, extractor = load_inference_bundle(status)
    preprocessor = TextPreprocessor()
    text = "Môi trường làm việc không được thân thiện, đồng nghiệp không hỗ trợ và ít cơ hội học hỏi."

    result = predict_review(text, model, extractor, preprocessor)

    assert result.label == "Negative"
    assert result.confidence is not None
    assert result.confidence >= 0.5
    assert result.decision_type in {"ml", "hybrid"}
    assert result.probabilities is not None
    assert result.probabilities["Negative"] > result.probabilities["Positive"]

