"""Interactive NLP workspace with real model probabilities and input diagnostics."""

from pathlib import Path
from time import perf_counter
import sys

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_services import (
    NEGATIVE_THRESHOLD, SENTIMENT_LABELS, SENTIMENT_ORDER,
    get_lexicon_model_status, get_model_status,
    load_inference_bundle, load_lexicon_inference_bundle,
    predict_review, predict_review_lexicon,
)
from src.app_theme import page_header, style_chart

EXAMPLES = {
    "Lời khen": "Môi trường làm việc thân thiện, đồng nghiệp hỗ trợ và có nhiều cơ hội học hỏi.",
    "Ý kiến hỗn hợp": "Công việc ổn, phúc lợi bình thường nhưng quy trình còn khá chậm.",
    "Lời phàn nàn": "Công ty rất tệ, quản lý yếu kém, lương thấp, thường xuyên bắt nhân viên OT không lương.",
    "Cần lưu ý": "Thường xuyên OT, quản lý thiếu minh bạch và lương chưa tương xứng.",
}
COLORS = {"Positive": "green", "Neutral": "yellow", "Negative": "red"}
MODEL_TEXT = "Text-only · 5.000 chiều"
MODEL_LEXICON = "Text + Lexicon · 5.005 chiều"


@st.cache_resource
def get_preprocessor():
    from src.preprocessing import TextPreprocessor
    return TextPreprocessor()


@st.cache_resource
def get_bundle(model_path: str, extractor_path: str):
    status = get_model_status()
    if not status.ready or str(status.model_path) != model_path:
        raise RuntimeError(status.message)
    if str(status.extractor_path) != extractor_path:
        raise RuntimeError("Feature extractor đã thay đổi trong lúc tải app.")
    return load_inference_bundle(status)


@st.cache_resource
def get_lexicon_bundle(model_path: str, extractor_path: str):
    status = get_lexicon_model_status()
    if not status.ready or str(status.model_path) != model_path:
        raise RuntimeError(status.message)
    if str(status.extractor_path) != extractor_path:
        raise RuntimeError("Lexicon extractor đã thay đổi trong lúc tải app.")
    return load_lexicon_inference_bundle(status)


def select_example():
    example = st.session_state.get("review_example")
    if example in EXAMPLES:
        text = EXAMPLES[example]
        st.session_state["review_input"] = text
        st.session_state["analysis_request"] = text
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_error", None)


def request_analysis():
    st.session_state["analysis_request"] = st.session_state.get("review_input", "")


def invalidate_result():
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_error", None)
    st.session_state.pop("analysis_request", None)


def render_pipeline_trace(cards: list[tuple[str, str, str, str, str]]) -> None:
    """Render six inference steps as three presentation-friendly stages."""
    stages = (
        (
            "text",
            "01",
            ":material/text_fields:",
            "Hiểu văn bản",
            "Từ review thô đến chuỗi token sạch",
            cards[:2],
        ),
        (
            "inference",
            "02",
            ":material/hub:",
            "Biểu diễn & dự đoán",
            "Biến token thành vector rồi tính xác suất",
            cards[2:4],
        ),
        (
            "decision",
            "03",
            ":material/check_circle:",
            "Ra quyết định",
            "Áp dụng quy tắc và trả nhãn cuối",
            cards[4:],
        ),
    )

    with st.container(horizontal=True, key="pipeline_stages", gap="small"):
        for stage_key, stage_number, icon, stage_title, stage_detail, stage_cards in stages:
            with st.container(
                border=True,
                height="content",
                key=f"pipeline_stage_{stage_key}",
            ):
                st.caption(f"GIAI ĐOẠN {stage_number}")
                st.markdown(f"### {icon} {stage_title}")
                st.caption(stage_detail)
                for key, step, title, value, detail in stage_cards:
                    with st.container(key=f"pipeline_step_{key}", height="content", gap="xsmall"):
                        st.caption(f"{step} / {title.upper()}")
                        st.markdown(f"**{value}**")
                        st.caption(detail)


def prediction_trace(original: str, model_name: str, prediction, elapsed: float):
    """Build presentation labels from the actual output of this inference run."""
    tokens = prediction.processed_text.split()
    top_terms = [term for term, _ in prediction.top_features[:3]]
    baseline_label = prediction.baseline_label or prediction.raw_ml_label or prediction.label
    baseline_vi = SENTIMENT_LABELS.get(baseline_label, baseline_label)
    baseline_probabilities = prediction.raw_ml_probabilities or dict(prediction.class_probabilities)
    baseline_score = baseline_probabilities.get(baseline_label)
    baseline_detail = (
        f"P({baseline_vi}) = {baseline_score:.1%} trước hiệu chỉnh."
        if baseline_score is not None
        else "Nhãn ban đầu trước bước hiệu chỉnh."
    )

    if prediction.decision_type == "hybrid":
        gate_value = "Hybrid Lexicon"
        gate_detail = f"Điều chỉnh {baseline_vi} → {SENTIMENT_LABELS.get(prediction.label, prediction.label)}."
    elif prediction.threshold_applied:
        gate_value = f"Threshold {NEGATIVE_THRESHOLD:.0%}"
        gate_detail = f"P(Tiêu cực) = {prediction.negative_probability:.1%}; quy tắc đổi nhãn."
    else:
        gate_value = "Không đổi nhãn"
        gate_detail = "Giữ nguyên dự đoán mặc định của Stacking."

    final_label = SENTIMENT_LABELS.get(prediction.label, prediction.label)
    confidence = f" · {prediction.confidence:.1%}" if prediction.confidence is not None else ""
    return [
        ("input", "01", "Review gốc", f"{len(original.strip()):,} ký tự", "Nội dung người dùng vừa cung cấp."),
        ("clean", "02", "Chuẩn hóa", f"{len(tokens):,} token", f"Mẫu: {', '.join(tokens[:4]) or 'không có token'}"),
        ("vector", "03", "Vector TF-IDF", f"{prediction.active_features:,} / {prediction.feature_count:,}", f"Top: {', '.join(top_terms) or 'không có đặc trưng'}"),
        ("model", "04", "Stacking model", baseline_vi, baseline_detail),
        ("gate", "05", "Hiệu chỉnh", gate_value, gate_detail),
        ("output", "06", "Nhãn cuối", f"{final_label}{confidence}", f"Hoàn tất trong {elapsed:.2f} giây · {model_name}"),
    ]


st.session_state.setdefault("review_input", "")
page_header(
    "PHÂN TÍCH CẢM XÚC", "Từ lời viết đến cảm xúc",
    "Nhập một review để nhận diện cảm xúc và theo dõi cách hệ thống đi đến kết quả.",
)
status = get_model_status()
lexicon_status = get_lexicon_model_status()
selected_model = st.segmented_control(
    "Mô hình suy luận",
    [MODEL_TEXT, MODEL_LEXICON],
    default=MODEL_TEXT,
    key="model_selector",
    on_change=invalidate_result,
    width="stretch",
    help="Text + Lexicon bổ sung 5 tín hiệu từ điển cảm xúc vào vector TF-IDF.",
)
use_lexicon_model = selected_model == MODEL_LEXICON
active_status = lexicon_status if use_lexicon_model else status
with st.container(horizontal=True, key="lab_status", gap="small"):
    st.badge("Model sẵn sàng" if active_status.ready else "Chờ model", color="green" if active_status.ready else "orange", icon=":material/memory:")
    st.badge("Stacking · TF-IDF + Lexicon" if use_lexicon_model else "Stacking · TF-IDF", color="blue")
    st.badge(f"Ngưỡng tiêu cực {NEGATIVE_THRESHOLD:.0%}", color="gray")

with st.container(key="prediction_workspace"):
    editor, result = st.columns([1.05, 1], gap="medium")

with editor.container(border=True, height="stretch", key="review_editor"):
    st.caption("01 / NỘI DUNG ĐẦU VÀO")
    st.subheader("Bạn muốn phân tích điều gì?")
    st.caption("Chọn một ví dụ để chạy ngay, hoặc viết review của bạn.")
    st.pills("Review mẫu", list(EXAMPLES), key="review_example", on_change=select_example,
             selection_mode="single", label_visibility="collapsed")
    review_text = st.text_area(
        "Nội dung review", key="review_input", height=200, max_chars=3000,
        placeholder="Ví dụ: Đồng nghiệp thân thiện nhưng công ty cần cải thiện chính sách OT…",
        on_change=invalidate_result, persist_state="session",
    )
    st.button("Phân tích cảm xúc", type="primary", icon=":material/arrow_forward:",
              on_click=request_analysis, width="stretch", key="analyze_review")
    st.caption("Chỉ dùng nội dung văn bản · Không cần rating hoặc tên công ty")
with result.container(border=True, height="stretch", key="prediction_result"):
    st.caption("02 / KẾT QUẢ PHÂN TÍCH")
    if "analysis_request" in st.session_state:
        requested = st.session_state.pop("analysis_request")
        invalidate_result()
        if not requested.strip():
            st.session_state["analysis_error"] = "Hãy nhập nội dung review trước khi phân tích."
        elif not active_status.ready or active_status.model_path is None:
            st.warning(active_status.message)
            st.code(get_preprocessor().clean_advance_text(requested), language=None)
        else:
            try:
                with st.spinner("Đang xử lý văn bản và tính xác suất…"):
                    started = perf_counter()
                    if use_lexicon_model:
                        model, extractor = get_lexicon_bundle(
                            str(lexicon_status.model_path), str(lexicon_status.extractor_path)
                        )
                        prediction = predict_review_lexicon(
                            requested, model, extractor, get_preprocessor()
                        )
                    else:
                        model, extractor = get_bundle(str(status.model_path), str(status.extractor_path))
                        prediction = predict_review(requested, model, extractor, get_preprocessor())
                    elapsed = perf_counter() - started
                st.session_state["analysis_result"] = (requested, selected_model, prediction, elapsed)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                st.session_state["analysis_error"] = str(exc)

    saved = st.session_state.get("analysis_result")
    if saved and (saved[0] != review_text or saved[1] != selected_model):
        saved = None
    error = st.session_state.get("analysis_error")
    if error:
        st.error(error, icon=":material/error:")
    elif saved:
        _, _, prediction, elapsed = saved
        label_vi = SENTIMENT_LABELS.get(prediction.label, prediction.label)
        st.subheader(f"Kết quả: {label_vi}")
        with st.container(horizontal=True, gap="small"):
            st.badge(label_vi, color=COLORS.get(prediction.label, "blue"))
            if prediction.decision_type == "hybrid":
                st.badge("Hybrid NLP + Lexicon", color="violet", icon=":material/auto_fix_high:")
            elif prediction.decision_type == "threshold":
                st.badge("Hiệu chỉnh ngưỡng", color="blue", icon=":material/tune:")
        for sentiment in SENTIMENT_ORDER:
            score = dict(prediction.class_probabilities).get(sentiment)
            if score is not None:
                with st.container(key=f"probability_{sentiment.lower()}", gap="xsmall"):
                    st.markdown(f"{SENTIMENT_LABELS[sentiment]} **{score:.1%}**")
                    st.progress(float(score))
        if not prediction.class_probabilities:
            st.caption("Model không cung cấp xác suất; chỉ hiển thị nhãn dự đoán.")
        if prediction.threshold_applied:
            baseline = SENTIMENT_LABELS.get(prediction.baseline_label, prediction.baseline_label)
            st.caption(f":material/tune: {baseline} → Tiêu cực vì P(Tiêu cực) = {prediction.negative_probability:.1%} ≥ {NEGATIVE_THRESHOLD:.0%}.")
        elif prediction.decision_type == "hybrid":
            baseline = SENTIMENT_LABELS.get(prediction.baseline_label, prediction.baseline_label)
            st.caption(f":material/auto_fix_high: Hybrid hiệu chỉnh từ {baseline} sang {label_vi} dựa trên tín hiệu từ điển và phủ định.")
        else:
            st.caption("Nhãn cuối trùng với dự đoán mặc định của model.")
        if prediction.explanation:
            st.info(prediction.explanation, icon=":material/info:")
        st.caption(f"Hoàn tất trong {elapsed:.2f} giây")
        if prediction.active_features == 0:
            st.warning("Văn bản không khớp từ điển TF-IDF. Kết quả này thiếu bằng chứng từ nội dung.")
    else:
        with st.container(key="prediction_empty"):
            st.markdown("## :material/neurology:")
            st.subheader("Một review, ba góc nhìn")
            st.write("Xác suất của từng lớp sẽ xuất hiện ở đây, cùng quy tắc chọn nhãn.")
            with st.container(horizontal=True, gap="small"):
                for sentiment in SENTIMENT_ORDER:
                    st.badge(SENTIMENT_LABELS[sentiment], color=COLORS[sentiment])
            st.caption("Bắt đầu bằng một ví dụ ở bên trái. Bạn cũng có thể thử câu khó hoặc ý kiến trái chiều.")

with st.container(border=True, key="nlp_evidence"):
    st.caption("03 / KHÁM PHÁ PIPELINE")
    st.subheader("Mô hình đã đọc review như thế nào?")
    if saved:
        original, model_name, prediction, elapsed = saved
        with st.container(horizontal=True, key="pipeline_trace_status", gap="small"):
            st.badge("6/6 bước hoàn tất", color="green", icon=":material/check_circle:")
            st.badge(model_name, color="blue", icon=":material/account_tree:")
        render_pipeline_trace(prediction_trace(original, model_name, prediction, elapsed))
        st.caption("Các giá trị được lấy trực tiếp từ lượt phân tích hiện tại; bước hiệu chỉnh được tách riêng khỏi dự đoán Stacking.")
        with st.container(horizontal=True, key="nlp_metrics", gap="small"):
            st.metric("Token sau xử lý", len(prediction.processed_text.split()))
            st.metric("Đặc trưng có giá trị", prediction.active_features)
            st.metric("Chiều vector đầu vào", f"{prediction.feature_count:,}")
        text_tab, vector_tab, decision_tab = st.tabs(["Văn bản & token", "Vector TF-IDF", "Quyết định & giới hạn"])
        with text_tab:
            st.caption("Unicode → emoji / teencode → tách từ → loại stopword")
            st.code(prediction.processed_text, language=None, wrap_lines=True)
            st.caption("Dấu gạch dưới nối các tiếng trong một từ ghép. Đây là đầu vào thực sự của TF-IDF.")
        with vector_tab:
            if prediction.top_features:
                st.caption("10 đặc trưng có trọng số TF-IDF cao nhất của review này. Trọng số đầu vào không phải mức đóng góp nhân quả vào nhãn.")
                features = pd.DataFrame(prediction.top_features, columns=["term", "weight"])
                bars = alt.Chart(features).mark_bar(color="#7aa7ff", cornerRadiusEnd=5, size=14).encode(
                    y=alt.Y("term:N", sort="-x", title=None),
                    x=alt.X("weight:Q", title="Trọng số TF-IDF"),
                    tooltip=[alt.Tooltip("term:N", title="Đặc trưng"), alt.Tooltip("weight:Q", title="Trọng số", format=".4f")],
                ).properties(height=240)
                st.altair_chart(style_chart(bars), width="stretch", theme=None)
            else:
                st.caption("Chưa có đặc trưng TF-IDF để hiển thị.")
        with decision_tab:
            st.write(f"**Quy tắc hiện tại:** nếu P(Tiêu cực) ≥ {NEGATIVE_THRESHOLD:.0%}, ưu tiên Tiêu cực; các trường hợp khác giữ nhãn mặc định.")
            if prediction.lexicon_stats:
                positive_phrases = prediction.lexicon_stats.get("pos_phrases", [])
                negative_phrases = prediction.lexicon_stats.get("neg_phrases", [])
                st.caption(
                    f"Tín hiệu từ điển · tích cực: {', '.join(positive_phrases[:4]) or 'không có'} · "
                    f"tiêu cực/phủ định: {', '.join(negative_phrases[:4]) or 'không có'}"
                )
            st.caption("Các số phần trăm là đầu ra của model, không phải cam kết độ chính xác cho từng review. Nhãn học từ rating có thể khác sắc thái thực tế của văn bản.")
            st.page_link("app_pages/benchmark.py", label="Xem mô hình và đánh giá chi tiết", icon=":material/arrow_forward:")
    else:
        render_pipeline_trace([
            ("input", "01", "Review gốc", "Nhận văn bản", "Chỉ cần nội dung review; không dùng rating."),
            ("clean", "02", "Chuẩn hóa", "Làm sạch tiếng Việt", "Unicode, emoji, teencode, từ ghép và stopword."),
            ("vector", "03", "Vector TF-IDF", "Biểu diễn đặc trưng", "Ánh xạ văn bản theo từ điển đã học."),
            ("model", "04", "Stacking model", "Tính xác suất", "Kết hợp các bộ phân loại đã khóa."),
            ("gate", "05", "Hiệu chỉnh", "Kiểm tra kết quả", f"Áp dụng ngưỡng Tiêu cực {NEGATIVE_THRESHOLD:.0%} và tín hiệu Lexicon nếu được chọn."),
            ("output", "06", "Nhãn cuối", "Trả kết quả", "Nhãn, độ tin cậy và bằng chứng của lượt chạy."),
        ])
        st.caption("Phân tích một review để xem token và trọng số thực tế ở từng bước.")

st.caption("NLP Sentiment Analysis · Kết quả nên được đọc cùng ngữ cảnh, đặc biệt với review vừa khen vừa chê.")
