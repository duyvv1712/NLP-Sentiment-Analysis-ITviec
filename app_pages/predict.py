"""Interactive review prediction page with a fail-closed model handoff."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_services import (
    SENTIMENT_LABELS,
    get_model_status,
    load_inference_bundle,
    predict_review,
)

if TYPE_CHECKING:
    from src.preprocessing import TextPreprocessor


EXAMPLES = {
    "Tích cực": "Môi trường làm việc thân thiện, đồng nghiệp hỗ trợ và có nhiều cơ hội học hỏi.",
    "Cân bằng": "Công việc ổn, phúc lợi bình thường nhưng quy trình còn khá chậm.",
    "Tiêu cực": "Thường xuyên OT, quản lý thiếu minh bạch và lương chưa tương xứng.",
}


@st.cache_resource
def get_preprocessor() -> TextPreprocessor:
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


st.title("Phân tích cảm xúc review")
st.caption("Nhập nội dung tự nhiên; hệ thống chỉ sử dụng văn bản, không dùng rating.")

status = get_model_status()
with st.container(border=True):
    with st.container(horizontal=True, vertical_alignment="center"):
        if status.ready:
            st.badge("Model sẵn sàng", icon=":material/check_circle:", color="green")
        else:
            st.badge("Chờ TV3 bàn giao model", icon=":material/schedule:", color="orange")
        st.caption(status.message)

st.subheader("Thử nhanh với review mẫu")
example = st.pills(
    "Review mẫu",
    options=list(EXAMPLES),
    selection_mode="single",
    key="review_example",
    label_visibility="collapsed",
)
if example and st.session_state.get("last_review_example") != example:
    st.session_state["review_input"] = EXAMPLES[example]
    st.session_state["last_review_example"] = example

with st.form("sentiment_form", border=True):
    review_text = st.text_area(
        "Nội dung review",
        key="review_input",
        placeholder="Ví dụ: Môi trường tốt nhưng công ty cần cải thiện chính sách OT...",
        height=170,
        max_chars=3000,
        help="Có thể nhập tiếng Việt, tiếng Anh ngành IT, teencode và emoji.",
    )
    with st.container(horizontal=True, horizontal_alignment="right"):
        submitted = st.form_submit_button(
            "Phân tích cảm xúc",
            type="primary",
            icon=":material/auto_awesome:",
        )

result_slot = st.container()
if submitted:
    if not review_text.strip():
        result_slot.error("Hãy nhập nội dung review trước khi phân tích.", icon=":material/error:")
    else:
        preprocessor = get_preprocessor()
        processed_preview = preprocessor.clean_advance_text(review_text)
        if not status.ready or status.model_path is None:
            with result_slot.container(border=True):
                st.warning(
                    "Giao diện đã sẵn sàng nhưng chưa thể dự đoán vì model chưa được bàn giao.",
                    icon=":material/pending:",
                )
                st.markdown("**Văn bản sau tiền xử lý**")
                st.code(processed_preview or "(không còn token hợp lệ)", language=None)
                st.caption("Không có nhãn hoặc điểm số giả được sinh ra trong trạng thái này.")
        else:
            try:
                model, extractor = get_bundle(
                    str(status.model_path), str(status.extractor_path)
                )
                prediction = predict_review(
                    review_text, model, extractor, preprocessor
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                result_slot.error(f"Không thể thực hiện dự đoán: {exc}", icon=":material/error:")
            else:
                label_vi = SENTIMENT_LABELS.get(prediction.label, prediction.label)
                with result_slot.container(border=True):
                    st.subheader(f"Kết quả: {label_vi}")
                    if prediction.confidence is None:
                        st.caption(
                            "Model không cung cấp xác suất đã hiệu chỉnh, vì vậy app không hiển thị confidence."
                        )
                    else:
                        st.metric("Độ tin cậy", f"{prediction.confidence:.1%}", border=True)
                    with st.expander("Xem văn bản sau tiền xử lý", icon=":material/code:"):
                        st.code(prediction.processed_text, language=None)

st.subheader("Pipeline suy luận")
with st.container(horizontal=True):
    with st.container(border=True):
        st.markdown("#### :material/chat: 1. Review")
        st.caption("Nội dung người dùng nhập, không yêu cầu rating hay thông tin công ty.")
    with st.container(border=True):
        st.markdown("#### :material/cleaning_services: 2. Tiền xử lý")
        st.caption("Chuẩn hóa Unicode, emoji, teencode, tách từ và loại stopword.")
    with st.container(border=True):
        st.markdown("#### :material/hub: 3. TF-IDF")
        st.caption("Biến văn bản thành vector 5.000 đặc trưng theo artifact đã khóa.")
    with st.container(border=True):
        st.markdown("#### :material/label: 4. Cảm xúc")
        st.caption("Trả về Tích cực, Trung tính hoặc Tiêu cực khi model sẵn sàng.")

with st.expander("Lưu ý khi diễn giải", icon=":material/info:"):
    st.markdown(
        """
        - Nhãn huấn luyện được suy ra từ rating nên không phải ground truth do con người gán trực tiếp.
        - Confidence chỉ hiển thị khi model có `predict_proba()`; app không tự chế điểm tin cậy.
        - Kết quả phục vụ demo học thuật, không thay thế đánh giá nhân sự chuyên môn.
        """
    )
