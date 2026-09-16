"""Unified model comparison, evaluation, and error analysis workspace."""

import hashlib
import json
from pathlib import Path
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_services import NEGATIVE_THRESHOLD, SENTIMENT_LABELS
from src.app_theme import page_header, style_chart


EVALUATION_DIR = PROJECT_ROOT / "reports" / "evaluation"
ABLATION_PATH = PROJECT_ROOT / "reports" / "aspect_hybrid_ablation.csv"

MODEL_SELECTION = pd.DataFrame(
    [
        {"rank": 1, "model": "Stacking Ensemble", "configuration": "NB + LR + Linear SVM", "cv_macro_f1": 0.5619},
        {"rank": 2, "model": "Logistic Regression", "configuration": "C=1.0 · L2 · balanced", "cv_macro_f1": 0.5567},
        {"rank": 3, "model": "Linear SVM", "configuration": "C=0.1 · balanced", "cv_macro_f1": 0.5561},
        {"rank": 4, "model": "Random Forest", "configuration": "200 cây · depth=20", "cv_macro_f1": 0.5515},
        {"rank": 5, "model": "Multinomial Naive Bayes", "configuration": "alpha=0.1", "cv_macro_f1": 0.4890},
    ]
)

FINAL_TEST_COMPARISON = pd.DataFrame(
    [
        {
            "model": "Stacking · Text-only",
            "training": "Supervised trên ITviec",
            "accuracy": 0.7766,
            "macro_f1": 0.5475,
            "negative_recall": 0.2632,
            "neutral_recall": 0.3476,
        },
        {
            "model": "ViSoBERT · Zero-shot",
            "training": "Chưa fine-tune ITviec",
            "accuracy": 0.6536,
            "macro_f1": 0.4036,
            "negative_recall": 0.7544,
            "neutral_recall": 0.0488,
        },
    ]
)


@st.cache_data(max_entries=2, show_spinner=False)
def load_evaluation_evidence(signature):
    """Load immutable evaluation outputs and verify the evaluated model hash."""
    snapshot = json.loads(
        (EVALUATION_DIR / "evaluation_snapshot.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(EVALUATION_DIR / "baseline_vs_negative_threshold.csv")
    sensitivity = pd.read_csv(EVALUATION_DIR / "negative_threshold_sensitivity.csv")
    errors = pd.read_csv(EVALUATION_DIR / "error_analysis_15_samples.csv")
    model_path = PROJECT_ROOT / "models" / "best_sentiment_model.joblib"
    model_matches = (
        model_path.exists()
        and hashlib.sha256(model_path.read_bytes()).hexdigest()
        == snapshot["model_sha256"]
    )
    return snapshot, comparison, sensitivity, errors, model_matches


@st.cache_data(max_entries=2, show_spinner=False)
def load_feature_ablation(signature):
    """Load deployable feature ablation results."""
    frame = pd.read_csv(ABLATION_PATH)
    return frame[frame["feature_group"].isin(["Text-only", "Text + lexicon"])].copy()


def confusion_rows(matrix: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Convert a confusion matrix to chart-ready rows."""
    return pd.DataFrame(
        [
            {
                "actual": SENTIMENT_LABELS[actual],
                "predicted": SENTIMENT_LABELS[predicted],
                "count": int(matrix[i, j]),
                "share": float(matrix[i, j] / matrix[i].sum()),
            }
            for i, actual in enumerate(labels)
            for j, predicted in enumerate(labels)
        ]
    )


def classification_report(matrix: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Derive per-class metrics directly from the displayed confusion matrix."""
    rows = []
    for index, label in enumerate(labels):
        true_positive = float(matrix[index, index])
        precision_denominator = float(matrix[:, index].sum())
        recall_denominator = float(matrix[index, :].sum())
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "Lớp cảm xúc": SENTIMENT_LABELS[label],
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Số mẫu": int(recall_denominator),
            }
        )
    return pd.DataFrame(rows)


page_header(
    "MÔ HÌNH & ĐÁNH GIÁ",
    "Từ lựa chọn mô hình đến phân tích lỗi",
    "So sánh công bằng, kiểm tra chất lượng theo từng lớp và quan sát tác động của ngưỡng Tiêu cực.",
)

view = st.segmented_control(
    "Nội dung phân tích",
    ["So sánh mô hình", "Chất lượng mô hình", "Lỗi & ngưỡng"],
    default="So sánh mô hình",
    key="model_evaluation_view",
    width="stretch",
)

try:
    evidence_paths = [
        EVALUATION_DIR / "evaluation_snapshot.json",
        EVALUATION_DIR / "baseline_vs_negative_threshold.csv",
        EVALUATION_DIR / "negative_threshold_sensitivity.csv",
        EVALUATION_DIR / "error_analysis_15_samples.csv",
        PROJECT_ROOT / "models" / "best_sentiment_model.joblib",
    ]
    evidence_signature = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in evidence_paths
    )
    snapshot, comparison, sensitivity, errors, model_matches = load_evaluation_evidence(
        evidence_signature
    )
    ablation_signature = (
        str(ABLATION_PATH),
        ABLATION_PATH.stat().st_mtime_ns,
        ABLATION_PATH.stat().st_size,
    )
    ablation = load_feature_ablation(ablation_signature)
except (OSError, ValueError, KeyError):
    st.warning(
        "Dữ liệu so sánh hoặc đánh giá chưa sẵn sàng. Vui lòng kiểm tra lại bộ tài nguyên của ứng dụng.",
        icon=":material/warning:",
    )
    st.stop()

baseline, policy = comparison.iloc[0], comparison.iloc[1]

with st.container(horizontal=True, key="evaluation_source_badges", gap="small"):
    st.badge("5-fold CV trên train", color="blue", icon=":material/cycle:")
    st.badge(
        f"Final test · {snapshot['test_count']:,} review",
        color="gray",
        icon=":material/verified:",
    )
    st.badge(
        "Model đúng phiên bản" if model_matches else "Model khác phiên bản đánh giá",
        color="green" if model_matches else "orange",
        icon=":material/fingerprint:",
    )

if not model_matches or not np.isclose(snapshot["threshold"], NEGATIVE_THRESHOLD):
    st.warning(
        "Mô hình hoặc ngưỡng hiện tại khác phiên bản đã tạo bộ chỉ số này."
    )


if view == "So sánh mô hình":
    st.caption("01 / LỰA CHỌN MÔ HÌNH")
    st.subheader("Stacking được chọn bằng cross-validation")

    with st.container(horizontal=True, key="comparison_summary", gap="small"):
        st.metric(
            "Mô hình được chọn",
            "Stacking",
            help="Ba base model: Multinomial Naive Bayes, Logistic Regression và Linear SVM.",
            border=True,
        )
        st.metric("CV Macro F1", "0.5619", border=True)
        st.metric("Final-test Macro F1", "0.5475", border=True)
        st.metric("ViSoBERT zero-shot", "0.4036", border=True)

    st.markdown("#### Bảng xếp hạng mô hình")
    leaderboard = MODEL_SELECTION.rename(
        columns={
            "rank": "Hạng",
            "model": "Mô hình",
            "configuration": "Cấu hình tốt nhất",
            "cv_macro_f1": "CV Macro F1",
        }
    )
    st.dataframe(
        leaderboard,
        hide_index=True,
        width="stretch",
        key="model_leaderboard",
        column_config={
            "Hạng": st.column_config.NumberColumn("Hạng", format="%d", width="small"),
            "Mô hình": st.column_config.TextColumn("Mô hình", pinned=True, width="medium"),
            "Cấu hình tốt nhất": st.column_config.TextColumn("Cấu hình tốt nhất", width="large"),
            "CV Macro F1": st.column_config.NumberColumn("CV Macro F1", format="%.4f", width="small"),
        },
    )

    st.markdown("#### Khoảng cách hiệu năng qua cross-validation")
    selection_chart = (
        alt.Chart(MODEL_SELECTION)
        .mark_bar(cornerRadiusEnd=7, size=24)
        .encode(
            x=alt.X(
                "cv_macro_f1:Q",
                title="CV Macro F1",
                scale=alt.Scale(domain=[0, 0.65]),
            ),
            y=alt.Y("model:N", title=None, sort=MODEL_SELECTION["model"].tolist()),
            color=alt.condition(
                alt.datum.rank == 1,
                alt.value("#50e3a4"),
                alt.value("#7aa7ff"),
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Mô hình"),
                alt.Tooltip("configuration:N", title="Cấu hình"),
                alt.Tooltip("cv_macro_f1:Q", title="CV Macro F1", format=".4f"),
            ],
        )
        .properties(height=250)
    )
    value_labels = selection_chart.mark_text(
        align="left", dx=6, color="#dbe7f7", font="JetBrains Mono", fontSize=12
    ).encode(text=alt.Text("cv_macro_f1:Q", format=".4f"))
    st.altair_chart(
        style_chart(selection_chart + value_labels), width="stretch", theme=None
    )
    st.caption(
        "Các cấu hình được xếp hạng bằng 5-fold cross-validation trên tập train; final test không được dùng để chọn model."
    )

    st.caption("02 / ĐÓNG GÓP CỦA LEXICON")
    st.subheader("Năm đặc trưng cảm xúc cải thiện nhẹ lớp khó")
    ablation_display = ablation.rename(
        columns={
            "feature_group": "Nhóm đặc trưng",
            "macro_f1": "Macro F1",
            "negative_f1": "F1 Tiêu cực",
            "negative_recall": "Recall Tiêu cực",
            "accuracy": "Accuracy",
        }
    )[["Nhóm đặc trưng", "Macro F1", "F1 Tiêu cực", "Recall Tiêu cực", "Accuracy"]]
    st.dataframe(
        ablation_display,
        hide_index=True,
        width="stretch",
        column_config={
            "Macro F1": st.column_config.NumberColumn(format="%.4f"),
            "F1 Tiêu cực": st.column_config.NumberColumn(format="%.4f"),
            "Recall Tiêu cực": st.column_config.NumberColumn(format="percent"),
            "Accuracy": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Đây là kết quả ablation qua cross-validation. Aspect rating không được đưa vào pipeline suy luận vì không có khi người dùng chỉ nhập văn bản."
    )

    st.caption("03 / ML TRUYỀN THỐNG VÀ TRANSFORMER")
    st.subheader("Model đúng miền vượt ViSoBERT chưa fine-tune")
    generalization_display = FINAL_TEST_COMPARISON.rename(
        columns={
            "model": "Mô hình",
            "training": "Cách sử dụng",
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "negative_recall": "Recall Tiêu cực",
            "neutral_recall": "Recall Trung tính",
        }
    )
    st.dataframe(
        generalization_display,
        hide_index=True,
        width="stretch",
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="percent"),
            "Macro F1": st.column_config.NumberColumn(format="%.4f"),
            "Recall Tiêu cực": st.column_config.NumberColumn(format="percent"),
            "Recall Trung tính": st.column_config.NumberColumn(format="percent"),
        },
    )
    with st.container(border=True, key="comparison_takeaway"):
        st.markdown("**Kết luận lựa chọn**")
        st.write(
            "Stacking được huấn luyện trực tiếp trên dữ liệu ITviec nên cân bằng ba lớp tốt hơn. "
            "ViSoBERT zero-shot bắt được nhiều review Tiêu cực nhưng gần như bỏ qua lớp Trung tính; "
            "đây là benchmark tham khảo, không phải so sánh với một Transformer đã fine-tune."
        )


elif view == "Chất lượng mô hình":
    st.caption("01 / KẾT QUẢ TRÊN FINAL TEST")
    st.subheader("Hiệu năng phải được đọc cùng lớp Tiêu cực")

    with st.container(horizontal=True, key="quality_metrics", gap="small"):
        for title, column, fmt in [
            ("Macro F1", "Macro F1", ".4f"),
            ("Accuracy", "Accuracy", ".1%"),
            ("Recall Tiêu cực", "Negative Recall", ".1%"),
            ("Precision Tiêu cực", "Negative Precision", ".1%"),
        ]:
            change = float(policy[column] - baseline[column])
            st.metric(
                title,
                format(policy[column], fmt),
                delta=(
                    f"{change:+.4f}"
                    if fmt == ".4f"
                    else f"{change * 100:+.2f} điểm %"
                ),
                border=True,
                help="Kết quả ở ngưỡng 30%; delta so với cách chọn nhãn mặc định.",
            )
    st.caption(
        "Các KPI đang hiển thị chính sách ngưỡng 30%; phần thay đổi được tính so với dự đoán argmax mặc định."
    )

    with st.container(border=True, key="quality_confusion_matrix"):
        st.caption("02 / MA TRẬN NHẦM LẪN")
        st.subheader("Model thường nhầm ở đâu?")
        matrix_mode = st.segmented_control(
            "Kết quả hiển thị",
            ["Mặc định", "Ngưỡng 30%"],
            default="Ngưỡng 30%",
            key="quality_matrix_mode",
        )
        matrix = np.asarray(
            snapshot[
                "baseline_matrix"
                if matrix_mode == "Mặc định"
                else "policy_matrix"
            ]
        )
        labels = snapshot["labels"]
        matrix_data = confusion_rows(matrix, labels)
        order = [SENTIMENT_LABELS[label] for label in labels]
        heat = alt.Chart(matrix_data).encode(
            x=alt.X(
                "predicted:N",
                title="Dự đoán",
                sort=order,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("actual:N", title="Nhãn từ rating", sort=order),
            tooltip=[
                alt.Tooltip("actual:N", title="Nhãn từ rating"),
                alt.Tooltip("predicted:N", title="Dự đoán"),
                alt.Tooltip("count:Q", title="Số review"),
                alt.Tooltip("share:Q", title="Tỷ lệ trong lớp thật", format=".1%"),
            ],
        )
        tiles = heat.mark_rect(
            cornerRadius=7, stroke="#0c1119", strokeWidth=5
        ).encode(
            color=alt.Color(
                "share:Q",
                scale=alt.Scale(domain=[0, 1], range=["#141f32", "#527bd1"]),
                legend=None,
            )
        )
        values = heat.mark_text(
            color="#f1f5f9", font="JetBrains Mono", fontSize=18
        ).encode(text="count:Q")
        st.altair_chart(
            style_chart((tiles + values).properties(height=260)),
            width="stretch",
            theme=None,
        )
        st.caption(
            "Mỗi hàng là một lớp thật. Màu được chuẩn hóa theo hàng để lớp ít mẫu vẫn nhìn thấy rõ."
        )

    st.caption("03 / CHỈ SỐ THEO TỪNG LỚP")
    st.subheader(f"Chi tiết · {matrix_mode}")
    class_report = classification_report(matrix, snapshot["labels"])
    st.dataframe(
        class_report,
        hide_index=True,
        width="stretch",
        column_config={
            "Precision": st.column_config.NumberColumn(format="percent"),
            "Recall": st.column_config.NumberColumn(format="percent"),
            "F1": st.column_config.NumberColumn(format="%.4f"),
            "Số mẫu": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.info(
        "Accuracy cao chủ yếu nhờ lớp Tích cực chiếm đa số. Macro F1 và Recall Tiêu cực phản ánh rõ hơn điểm yếu của model.",
        icon=":material/info:",
    )


else:
    st.caption("01 / ĐÁNH ĐỔI KHI HẠ NGƯỠNG")
    st.subheader("Bắt được nhiều hơn, cảnh báo nhầm cũng nhiều hơn")
    base_matrix = np.asarray(snapshot["baseline_matrix"])
    policy_matrix = np.asarray(snapshot["policy_matrix"])
    negative_index = snapshot["labels"].index("Negative")
    recovered = int(
        policy_matrix[negative_index, negative_index]
        - base_matrix[negative_index, negative_index]
    )
    added_false_positive = int(
        (
            policy_matrix[:, negative_index].sum()
            - policy_matrix[negative_index, negative_index]
        )
        - (
            base_matrix[:, negative_index].sum()
            - base_matrix[negative_index, negative_index]
        )
    )
    with st.container(horizontal=True, key="threshold_tradeoff", gap="small"):
        st.metric("Tiêu cực nhận đúng thêm", f"+{recovered}", border=True)
        st.metric(
            "Cảnh báo nhầm Tiêu cực thêm",
            f"+{added_false_positive}",
            border=True,
        )
        st.metric(
            "Recall Tiêu cực",
            f"{policy['Negative Recall']:.1%}",
            delta=f"{(policy['Negative Recall'] - baseline['Negative Recall']) * 100:+.2f} điểm %",
            border=True,
        )
        st.metric(
            "Precision Tiêu cực",
            f"{policy['Negative Precision']:.1%}",
            delta=f"{(policy['Negative Precision'] - baseline['Negative Precision']) * 100:+.2f} điểm %",
            border=True,
        )
    st.caption(
        "Threshold chỉ thay đổi quy tắc chọn nhãn sau model; trọng số Stacking và TF-IDF được giữ nguyên."
    )

    with st.container(border=True, key="threshold_sensitivity"):
        st.caption("02 / KHÁM PHÁ NGƯỠNG")
        st.subheader("Recall và Precision thay đổi như thế nào?")
        selected_threshold = st.select_slider(
            "Ngưỡng đang khảo sát",
            options=sensitivity["Threshold"].tolist(),
            value=0.30,
            key="evaluation_threshold",
        )
        selected = sensitivity.loc[
            np.isclose(sensitivity["Threshold"], selected_threshold)
        ].iloc[0]
        chart_data = sensitivity.rename(
            columns={
                "Negative Recall": "Recall Tiêu cực",
                "Negative Precision": "Precision Tiêu cực",
                "Macro F1": "Macro F1",
            }
        ).melt(
            id_vars=["Threshold"],
            value_vars=["Recall Tiêu cực", "Precision Tiêu cực", "Macro F1"],
            var_name="metric",
            value_name="value",
        )
        lines = (
            alt.Chart(chart_data)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X(
                    "Threshold:Q",
                    title="Ngưỡng Tiêu cực",
                    axis=alt.Axis(format=".0%"),
                ),
                y=alt.Y(
                    "value:Q",
                    title=None,
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%"),
                ),
                color=alt.Color(
                    "metric:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Recall Tiêu cực", "Precision Tiêu cực", "Macro F1"],
                        range=["#50e3a4", "#ff994f", "#7aa7ff"],
                    ),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("Threshold:Q", title="Ngưỡng", format=".0%"),
                    alt.Tooltip("metric:N", title="Chỉ số"),
                    alt.Tooltip("value:Q", title="Giá trị", format=".2%"),
                ],
            )
        )
        marker = alt.Chart(
            pd.DataFrame({"Threshold": [selected_threshold]})
        ).mark_rule(strokeDash=[4, 4], color="#cdd8e8").encode(x="Threshold:Q")
        st.altair_chart(
            style_chart((lines + marker).properties(height=250)),
            width="stretch",
            theme=None,
        )
        st.caption(
            f"Ngưỡng {selected_threshold:.0%} · Recall {selected['Negative Recall']:.2%} · "
            f"Precision {selected['Negative Precision']:.2%} · Macro F1 {selected['Macro F1']:.4f}"
        )
        st.caption(
            "Thanh trượt chỉ dùng để xem số liệu đã lưu; nó không thay đổi ngưỡng đang chạy trong trang Phân tích review."
        )

    with st.container(border=True, key="evaluation_errors"):
        st.caption("03 / ERROR ANALYSIS")
        st.subheader("Đọc một lỗi thật để hiểu giới hạn")
        pairs = errors["Cặp nhầm"].unique().tolist()
        pair = st.selectbox(
            "Kiểu nhầm lẫn",
            pairs,
            format_func=lambda value: " → ".join(
                SENTIMENT_LABELS.get(part, part) for part in value.split(" → ")
            ),
            key="error_pair",
        )
        subset = errors[errors["Cặp nhầm"] == pair]
        error_index = st.selectbox(
            "Review cần xem",
            subset.index.tolist(),
            format_func=lambda index: (
                f"#{int(errors.loc[index, 'source_index'])} · "
                f"{errors.loc[index, 'Company Name']} · "
                f"{int(errors.loc[index, 'Rating'])} sao"
            ),
            key="error_review",
        )
        row = errors.loc[error_index]
        with st.container(horizontal=True, key="error_badges", gap="small"):
            st.badge(
                f"Nhãn từ rating: {SENTIMENT_LABELS[row['Nhãn thật']]}",
                color="gray",
            )
            st.badge(
                f"Model: {SENTIMENT_LABELS[row['Threshold 0.30']]}",
                color="orange",
            )
            st.badge(f"P(Tiêu cực): {row['P(Negative)']:.1%}", color="gray")
        with st.container(height=190, border=True):
            st.text(row["raw_review_text"])
        st.caption(
            f"Tín hiệu tham khảo: {row['Nhóm tín hiệu']}. Nhãn được suy ra từ rating nên có thể khác sắc thái của từng câu trong review."
        )
        st.caption(
            "Các ví dụ giúp giải thích kiểu lỗi; tỷ lệ tổng thể phải được đọc ở ma trận nhầm lẫn."
        )

    st.warning(
        "Ngưỡng 30% chưa đạt mục tiêu Recall 55–60% và hiện là phân tích post-hoc trên final test đã đánh giá.",
        icon=":material/science:",
    )
