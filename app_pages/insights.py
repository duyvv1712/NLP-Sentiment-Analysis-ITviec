"""Interactive company sentiment dashboard backed by cleaned ITviec reviews."""

from io import BytesIO
from pathlib import Path
import sys

import altair as alt
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_cache import get_reviews
from src.app_services import (
    SENTIMENT_ORDER,
    company_summary,
    monthly_sentiment,
    sentiment_summary,
    top_terms,
)


@st.cache_data(max_entries=48, show_spinner="Đang tạo WordCloud...")
def make_wordcloud(text: str, sentiment: str, theme_type: str) -> bytes:
    from wordcloud import WordCloud

    if theme_type == "dark":
        background_color = "#121720"
        color_map = "summer" if sentiment == "Positive" else "autumn"
    else:
        background_color = "white"
        color_map = "Greens" if sentiment == "Positive" else "Reds"
    image = WordCloud(
        width=1200,
        height=520,
        background_color=background_color,
        max_words=120,
        colormap=color_map,
        collocations=False,
        random_state=2026,
    ).generate(text)
    buffer = BytesIO()
    image.to_image().save(buffer, format="PNG")
    return buffer.getvalue()


st.title("Insight cảm xúc doanh nghiệp")
st.caption(
    "Khám phá xu hướng, từ khóa và review thật. Bộ lọc ngưỡng mẫu giúp tránh "
    "so sánh doanh nghiệp có quá ít dữ liệu."
)

try:
    reviews = get_reviews()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

with st.container(border=True, key="insight_filter_panel"):
    st.markdown("**:material/tune: Bộ lọc phân tích**")
    min_reviews = st.segmented_control(
        "Ngưỡng mẫu tối thiểu",
        options=[30, 50, 100, 200],
        default=50,
        format_func=lambda value: f"{value}+ review",
        key="minimum_reviews",
    )
    threshold = int(min_reviews or 50)
    companies = company_summary(reviews, min_reviews=threshold)
    company_options = ["Toàn bộ dữ liệu", *companies["Company Name"].tolist()]
    selected_company = st.selectbox(
        "Doanh nghiệp",
        options=company_options,
        key="selected_company",
        help="Chỉ các doanh nghiệp đạt ngưỡng mẫu mới xuất hiện.",
    )

selected_reviews = (
    reviews
    if selected_company == "Toàn bộ dữ liệu"
    else reviews[reviews["Company Name"] == selected_company]
)
summary = sentiment_summary(selected_reviews)
positive_share = float(summary.loc[summary["sentiment"] == "Positive", "share"].iloc[0])
neutral_share = float(summary.loc[summary["sentiment"] == "Neutral", "share"].iloc[0])
negative_share = float(summary.loc[summary["sentiment"] == "Negative", "share"].iloc[0])

with st.container(horizontal=True, gap="xsmall"):
    with st.container(key="kpi_reviews"):
        st.metric("Review", f"{len(selected_reviews):,}", icon=":material/rate_review:", border=True)
    with st.container(key="kpi_positive"):
        st.metric("Tích cực", f"{positive_share:.1f}%", icon=":material/sentiment_satisfied:", border=True)
    with st.container(key="kpi_neutral"):
        st.metric("Trung tính", f"{neutral_share:.1f}%", icon=":material/sentiment_neutral:", border=True)
    with st.container(key="kpi_negative"):
        st.metric("Tiêu cực", f"{negative_share:.1f}%", icon=":material/sentiment_dissatisfied:", border=True)
    with st.container(key="kpi_rating"):
        st.metric("Điểm trung bình", f"{selected_reviews['Rating'].mean():.2f}/5", icon=":material/star:", border=True)

left, right = st.columns([0.9, 1.4], gap="large")
with left:
    with st.container(border=True, height="stretch", key="insight_donut_panel"):
        st.subheader(":material/donut_large: Cơ cấu cảm xúc")
        donut = (
            alt.Chart(summary)
            .mark_arc(innerRadius=68, outerRadius=118, cornerRadius=5)
            .encode(
                theta=alt.Theta("reviews:Q"),
                color=alt.Color(
                    "sentiment:N",
                    scale=alt.Scale(domain=list(SENTIMENT_ORDER)),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Cảm xúc"),
                    alt.Tooltip("reviews:Q", title="Review", format=","),
                    alt.Tooltip("share:Q", title="Tỷ lệ", format=".1f"),
                ],
            )
            .properties(height=330)
        )
        st.altair_chart(donut, width="stretch")

with right:
    with st.container(border=True, height="stretch", key="insight_trend_panel"):
        st.subheader(":material/timeline: Xu hướng theo thời gian")
        trend = monthly_sentiment(selected_reviews)
        if trend.empty:
            st.info("Không đủ dữ liệu thời gian để vẽ xu hướng.", icon=":material/info:")
        else:
            line = (
                alt.Chart(trend)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X("review_month:T", title=None, axis=alt.Axis(format="%m/%Y")),
                    y=alt.Y("reviews:Q", title="Số review"),
                    color=alt.Color(
                        "sentiment:N",
                        scale=alt.Scale(domain=list(SENTIMENT_ORDER)),
                        legend=alt.Legend(title="Cảm xúc", orient="bottom"),
                    ),
                    tooltip=[
                        alt.Tooltip("yearmonth(review_month):T", title="Tháng"),
                        alt.Tooltip("sentiment:N", title="Cảm xúc"),
                        alt.Tooltip("reviews:Q", title="Review"),
                    ],
                )
                .properties(height=330)
                .interactive()
            )
            st.altair_chart(line, width="stretch")

st.subheader("Ngôn ngữ nổi bật")
language_expander = st.expander(
    "Mở WordCloud và từ khóa",
    icon=":material/cloud:",
    on_change="rerun",
)
if language_expander.open:
    with language_expander:
        tone = st.segmented_control(
            "Nhóm cảm xúc",
            options=["Positive", "Negative"],
            default="Positive",
            format_func=lambda value: (
                "Tích cực" if value == "Positive" else "Tiêu cực"
            ),
            key="wordcloud_tone",
        )
        selected_tone = tone or "Positive"
        tone_reviews = selected_reviews[
            selected_reviews["sentiment"] == selected_tone
        ]
        clean_text = " ".join(
            tone_reviews["clean_advance_text"].dropna().astype(str)
        )
        theme_type = "dark" if st.context.theme.type == "dark" else "light"

        cloud_col, terms_col = st.columns([1.5, 0.8], gap="large")
        with cloud_col:
            with st.container(border=True, height="stretch", key="wordcloud_panel"):
                st.markdown(
                    "**WordCloud tích cực**"
                    if selected_tone == "Positive"
                    else "**WordCloud tiêu cực**"
                )
                if clean_text.strip():
                    st.image(
                        make_wordcloud(clean_text, selected_tone, theme_type),
                        width="stretch",
                    )
                else:
                    st.info(
                        "Không có nội dung phù hợp để tạo WordCloud.",
                        icon=":material/info:",
                    )

        with terms_col:
            with st.container(border=True, height="stretch", key="terms_panel"):
                st.markdown("**15 từ khóa xuất hiện nhiều**")
                terms = top_terms(tone_reviews["clean_advance_text"], limit=15)
                if terms.empty:
                    st.caption("Chưa có từ khóa.")
                else:
                    terms = terms.assign(sentiment=selected_tone)
                    term_chart = (
                        alt.Chart(terms)
                        .mark_bar(cornerRadiusEnd=5)
                        .encode(
                            x=alt.X("count:Q", title="Số lần"),
                            y=alt.Y("term:N", title=None, sort="-x"),
                            color=alt.Color(
                                "sentiment:N",
                                scale=alt.Scale(domain=list(SENTIMENT_ORDER)),
                                legend=None,
                            ),
                            tooltip=["term:N", "count:Q"],
                        )
                        .properties(height=390)
                    )
                    st.altair_chart(term_chart, width="stretch")
else:
    st.caption(
        "WordCloud chỉ được tạo khi bạn mở phần này để giữ dashboard phản hồi nhanh."
    )

st.subheader("So sánh doanh nghiệp đủ ngưỡng mẫu")
with st.container(border=True, key="company_table_panel"):
    st.caption(
        f"Có {len(companies)} doanh nghiệp đạt ngưỡng {threshold}+ review. "
        "Bảng được sắp xếp theo quy mô mẫu, không phải bảng xếp hạng nơi làm việc."
    )
    display_companies = companies.rename(
        columns={
            "Company Name": "Doanh nghiệp",
            "reviews": "Số review",
            "average_rating": "Rating TB",
            "positive_share": "Tích cực",
            "neutral_share": "Trung tính",
            "negative_share": "Tiêu cực",
        }
    )[
        [
            "Doanh nghiệp",
            "Số review",
            "Rating TB",
            "Tích cực",
            "Trung tính",
            "Tiêu cực",
        ]
    ]
    st.dataframe(
        display_companies,
        hide_index=True,
        column_config={
            "Doanh nghiệp": st.column_config.TextColumn(pinned=True),
            "Số review": st.column_config.NumberColumn(format="%d"),
            "Rating TB": st.column_config.NumberColumn(format="%.2f"),
            "Tích cực": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100
            ),
            "Trung tính": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100
            ),
            "Tiêu cực": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=100
            ),
        },
        height=360,
    )

review_expander = st.expander(
    "Khám phá review chi tiết", icon=":material/rate_review:", on_change="rerun"
)
if review_expander.open:
    with review_expander:
        selected_sentiments = st.pills(
            "Cảm xúc",
            options=list(SENTIMENT_ORDER),
            default=list(SENTIMENT_ORDER),
            selection_mode="multi",
            format_func=lambda value: {
                "Positive": "Tích cực",
                "Neutral": "Trung tính",
                "Negative": "Tiêu cực",
            }[value],
            key="review_sentiments",
        )
        filtered_reviews = selected_reviews[
            selected_reviews["sentiment"].isin(selected_sentiments or [])
        ]
        review_columns = [
            column
            for column in [
                "Title",
                "What I liked",
                "Suggestions for improvement",
                "Rating",
                "sentiment",
                "Cmt_day",
            ]
            if column in filtered_reviews.columns
        ]
        st.dataframe(
            filtered_reviews[review_columns].head(200),
            hide_index=True,
            column_config={
                "Title": st.column_config.TextColumn("Tiêu đề", width="medium"),
                "What I liked": st.column_config.TextColumn("Điểm thích", width="large"),
                "Suggestions for improvement": st.column_config.TextColumn(
                    "Gợi ý cải thiện", width="large"
                ),
                "Rating": st.column_config.NumberColumn("Rating", format="%d ★"),
                "sentiment": st.column_config.TextColumn("Cảm xúc"),
                "Cmt_day": st.column_config.TextColumn("Thời gian"),
            },
            height=420,
        )

st.caption(
    "Lưu ý: sentiment trong dashboard là weak label suy ra từ rating; phân tích này mô tả dữ liệu, "
    "không khẳng định quan hệ nhân quả."
)
