"""Shared dark developer-tool styling for the Streamlit application."""

from __future__ import annotations

import streamlit as st


def apply_app_style() -> None:
    """Apply the compact glass surface system around native Streamlit widgets."""
    st.html(
        """
        <style>
        :root {
          --it-bg: #080b10;
          --it-surface: rgba(18, 23, 32, .78);
          --it-surface-strong: #141a24;
          --it-surface-soft: rgba(255, 255, 255, .035);
          --it-border: rgba(255, 255, 255, .085);
          --it-border-hover: rgba(255, 255, 255, .16);
          --it-text: #f1f5f9;
          --it-muted: #8691a2;
          --it-muted-2: #5d6776;
          --it-blue: #7aa7ff;
          --it-mint: #50e3a4;
          --it-warning: #f4d35e;
          --it-orange: #ff994f;
          --it-error: #ff6677;
          --it-radius-sm: 11px;
          --it-radius-md: 15px;
          --it-radius-lg: 18px;
          --it-font: "DM Sans", system-ui, sans-serif;
          --it-mono: "JetBrains Mono", ui-monospace, monospace;
        }

        html, body, [data-testid="stAppViewContainer"] {
          font-family: var(--it-font);
        }

        /* A readable dashboard canvas for laptops and 34-inch ultrawide screens. */
        [data-testid="stMainBlockContainer"] {
          width: 100%;
          max-width: 1920px;
          margin-inline: auto;
          padding-inline: clamp(1rem, 3vw, 4rem);
          padding-top: 4rem;
          padding-bottom: 3rem;
        }
        [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(circle at 78% -12%, rgba(122, 167, 255, .10) 0, transparent 31rem),
            radial-gradient(circle at 8% 105%, rgba(80, 227, 164, .055) 0, transparent 28rem),
            var(--it-bg);
        }
        header[data-testid="stHeader"] {
          background: rgba(8, 11, 16, .74);
          backdrop-filter: blur(10px);
        }
        [data-testid="stMainMenuPopover"] {
          opacity: 0 !important;
          pointer-events: none !important;
        }

        /* Sidebar remains calm, compact, and fully collapsible. */
        section[data-testid="stSidebar"] {
          background: rgba(12, 16, 23, .94);
          border-right: 1px solid var(--it-border);
          box-shadow: 10px 0 32px rgba(0, 0, 0, .14);
        }
        [data-testid="stSidebarContent"] {
          padding-top: .8rem;
        }
        [data-testid="stSidebarNavLink"] {
          margin-block: 2px;
          border: 1px solid transparent;
          border-radius: var(--it-radius-sm);
          transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
        }
        [data-testid="stSidebarNavLink"]:hover {
          transform: translateX(2px);
          background: rgba(255, 255, 255, .04);
          border-color: var(--it-border);
        }
        [data-testid="stSidebarNavLink"][aria-current="page"] {
          color: var(--it-text);
          background: rgba(122, 167, 255, .11);
          border-color: rgba(122, 167, 255, .18);
          box-shadow: inset 2px 0 0 var(--it-blue);
        }

        /* Shared restrained glass surfaces. */
        [data-testid="stMetric"],
        [data-testid="stVerticalBlockBorderWrapper"] {
          border: 1px solid var(--it-border);
          background: var(--it-surface);
          backdrop-filter: blur(12px);
          box-shadow: 0 12px 32px rgba(0, 0, 0, .16);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
          border-radius: var(--it-radius-lg);
          transition: border-color 150ms ease, box-shadow 150ms ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
          border-color: var(--it-border-hover);
          box-shadow: 0 14px 36px rgba(0, 0, 0, .19);
        }
        [data-testid="stMetric"] {
          min-height: 112px;
          padding: .9rem 1rem;
          border-radius: var(--it-radius-md);
          transition: border-color 150ms ease, transform 150ms ease;
        }
        [data-testid="stMetric"]:hover {
          transform: translateY(-2px);
          border-color: rgba(122, 167, 255, .26);
        }
        :is(
          .st-key-kpi_reviews,
          .st-key-kpi_companies,
          .st-key-kpi_positive,
          .st-key-kpi_neutral,
          .st-key-kpi_negative,
          .st-key-kpi_rating
        ) {
          --metric-accent: var(--it-blue);
        }
        .st-key-kpi_positive { --metric-accent: var(--it-mint); }
        .st-key-kpi_neutral { --metric-accent: var(--it-warning); }
        .st-key-kpi_negative { --metric-accent: var(--it-error); }
        .st-key-kpi_rating { --metric-accent: var(--it-orange); }
        :is(
          .st-key-kpi_reviews,
          .st-key-kpi_companies,
          .st-key-kpi_positive,
          .st-key-kpi_neutral,
          .st-key-kpi_negative,
          .st-key-kpi_rating
        ) [data-testid="stMetric"] {
          position: relative;
          overflow: hidden;
        }
        :is(
          .st-key-kpi_reviews,
          .st-key-kpi_companies,
          .st-key-kpi_positive,
          .st-key-kpi_neutral,
          .st-key-kpi_negative,
          .st-key-kpi_rating
        ) [data-testid="stMetric"]::before {
          content: "";
          position: absolute;
          inset: 0 auto 0 0;
          width: 2px;
          background: var(--metric-accent);
          opacity: .78;
        }
        :is(
          .st-key-kpi_reviews,
          .st-key-kpi_companies,
          .st-key-kpi_positive,
          .st-key-kpi_neutral,
          .st-key-kpi_negative,
          .st-key-kpi_rating
        ) :is([data-testid="stMetricValue"], [data-testid="stMetricLabel"] [data-testid="stIconMaterial"]) {
          color: var(--metric-accent);
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
          font-family: var(--it-mono);
        }
        [data-testid="stMetricLabel"] {
          color: var(--it-muted);
          font-size: .72rem;
          letter-spacing: .035em;
        }
        [data-testid="stMetricValue"] {
          color: var(--it-blue);
          font-size: 1.72rem;
          letter-spacing: -.045em;
        }

        /* Inputs and compact selectors. */
        [data-testid="stWidgetLabel"],
        [data-testid="stButtonGroup"] button,
        [data-testid="stBadge"],
        code, pre, kbd {
          font-family: var(--it-mono);
        }
        [data-testid="stWidgetLabel"] {
          color: #aab5c6;
          font-size: .72rem;
          letter-spacing: .025em;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {
          border-color: var(--it-border);
          border-radius: var(--it-radius-sm);
          background: rgba(6, 9, 14, .66);
          transition: border-color 140ms ease, box-shadow 140ms ease;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
        [data-testid="stTextArea"] textarea:focus,
        [data-testid="stTextInput"] input:focus {
          border-color: rgba(122, 167, 255, .52);
          box-shadow: 0 0 0 3px rgba(122, 167, 255, .08);
        }
        [data-testid="stButtonGroup"] [role="toolbar"],
        [data-testid="stButtonGroup"] [role="radiogroup"] {
          gap: .24rem;
          padding: .25rem;
          border: 1px solid var(--it-border);
          border-radius: var(--it-radius-sm);
          background: rgba(5, 8, 13, .62);
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, .025);
        }
        [data-testid="stButtonGroup"] button[data-variant="segmented_control"],
        [data-testid="stButtonGroup"] button[data-variant="pills"] {
          min-height: 36px;
          margin: 0;
          border: 1px solid transparent;
          border-radius: 8px !important;
          color: var(--it-muted);
          font-size: .68rem;
          transition: color 140ms ease, background 140ms ease, border-color 140ms ease;
        }
        [data-testid="stButtonGroup"] button:hover {
          color: var(--it-text);
          background: rgba(255, 255, 255, .045);
        }
        [data-testid="stButtonGroup"] button[data-selected] {
          color: #e9f2ff;
          border-color: rgba(122, 167, 255, .20);
          background: rgba(122, 167, 255, .12);
          box-shadow: 0 3px 10px rgba(20, 45, 82, .14);
        }

        /* Actions use solid color rather than bright gradients. */
        [data-testid="stBaseButton-primary"],
        [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
          min-height: 40px;
          padding-inline: 1.1rem;
          border: 1px solid rgba(122, 167, 255, .34);
          border-radius: var(--it-radius-sm);
          color: #08101d;
          background: var(--it-blue);
          box-shadow: 0 7px 18px rgba(56, 92, 154, .16);
          font-weight: 700;
          transition: transform 140ms ease, filter 140ms ease, box-shadow 140ms ease;
        }
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover {
          transform: translateY(-1px);
          filter: brightness(1.05);
          box-shadow: 0 9px 22px rgba(56, 92, 154, .20);
        }
        [data-testid="stBaseButton-secondary"],
        [data-testid="stPopoverButton"] > button {
          min-height: 40px;
          border-color: var(--it-border);
          border-radius: var(--it-radius-sm);
          color: var(--it-text);
          background: rgba(255, 255, 255, .045);
          box-shadow: none;
          transition: transform 140ms ease, background 140ms ease, border-color 140ms ease;
        }
        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stPopoverButton"] > button:hover {
          transform: translateY(-1px);
          border-color: var(--it-border-hover);
          background: rgba(255, 255, 255, .07);
        }

        /* Tabs, expanders, alerts, tables, and status elements. */
        [data-testid="stTabs"] [role="tablist"] {
          gap: .25rem;
          padding: .25rem;
          border: 1px solid var(--it-border);
          border-radius: var(--it-radius-sm);
          background: rgba(5, 8, 13, .58);
        }
        [data-testid="stTabs"] [role="tab"] {
          min-height: 38px;
          padding-inline: .9rem;
          border-radius: 8px;
          color: var(--it-muted);
          transition: background 140ms ease, color 140ms ease;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
          color: var(--it-text);
          background: rgba(122, 167, 255, .11);
        }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"],
        [data-testid="stTabs"] [data-baseweb="tab-border"] {
          display: none;
        }
        [data-testid="stExpander"] details {
          border-color: var(--it-border);
          border-radius: var(--it-radius-md);
          background: var(--it-surface);
        }
        [data-testid="stAlert"] {
          border-radius: var(--it-radius-sm);
          border-color: var(--it-border);
          background: rgba(18, 23, 32, .72);
        }
        [data-testid="stDataFrame"] {
          overflow: hidden;
          border: 1px solid var(--it-border);
          border-radius: var(--it-radius-md);
        }
        [data-testid="stBadge"] {
          border: 1px solid rgba(80, 227, 164, .16);
          color: var(--it-mint);
          background: rgba(80, 227, 164, .08);
          font-size: .65rem;
          letter-spacing: .025em;
        }
        [data-testid="stCaptionContainer"] {
          color: var(--it-muted);
        }

        /* Named sections create a clearer hierarchy without heavier glass. */
        :is(
          .st-key-overview_sentiment_panel,
          .st-key-overview_journey_panel,
          .st-key-insight_filter_panel,
          .st-key-insight_donut_panel,
          .st-key-insight_trend_panel,
          .st-key-wordcloud_panel,
          .st-key-terms_panel,
          .st-key-company_table_panel,
          .st-key-model_status_panel,
          .st-key-sentiment_form,
          .st-key-product_data_card,
          .st-key-product_insight_card,
          .st-key-product_model_card,
          .st-key-pipeline_review_card,
          .st-key-pipeline_clean_card,
          .st-key-pipeline_vector_card,
          .st-key-pipeline_result_card
        ) {
          position: relative;
          overflow: hidden;
          border-color: var(--it-border) !important;
          background: var(--it-surface);
          backdrop-filter: blur(12px);
          box-shadow: 0 12px 32px rgba(0, 0, 0, .15);
        }
        :is(
          .st-key-overview_sentiment_panel,
          .st-key-overview_journey_panel,
          .st-key-insight_filter_panel,
          .st-key-insight_donut_panel,
          .st-key-insight_trend_panel,
          .st-key-wordcloud_panel,
          .st-key-terms_panel,
          .st-key-company_table_panel,
          .st-key-model_status_panel,
          .st-key-sentiment_form,
          .st-key-product_data_card,
          .st-key-product_insight_card,
          .st-key-product_model_card,
          .st-key-pipeline_review_card,
          .st-key-pipeline_clean_card,
          .st-key-pipeline_vector_card,
          .st-key-pipeline_result_card
        )::before {
          content: "";
          position: absolute;
          z-index: 1;
          top: 0;
          left: 18px;
          width: 42px;
          height: 2px;
          border-radius: 0 0 2px 2px;
          background: var(--card-accent, var(--it-blue));
          opacity: .72;
        }
        .st-key-product_data_card,
        .st-key-pipeline_clean_card { --card-accent: var(--it-mint); }
        .st-key-product_insight_card,
        .st-key-pipeline_vector_card { --card-accent: var(--it-blue); }
        .st-key-product_model_card,
        .st-key-pipeline_result_card { --card-accent: var(--it-warning); }
        .st-key-pipeline_review_card { --card-accent: var(--it-orange); }
        :is(
          .st-key-product_data_card,
          .st-key-product_insight_card,
          .st-key-product_model_card,
          .st-key-pipeline_review_card,
          .st-key-pipeline_clean_card,
          .st-key-pipeline_vector_card,
          .st-key-pipeline_result_card
        ) {
          min-height: 138px;
          transition: transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
        }
        :is(
          .st-key-product_data_card,
          .st-key-product_insight_card,
          .st-key-product_model_card,
          .st-key-pipeline_review_card,
          .st-key-pipeline_clean_card,
          .st-key-pipeline_vector_card,
          .st-key-pipeline_result_card
        ):hover {
          transform: translateY(-2px);
          border-color: color-mix(in srgb, var(--card-accent) 28%, transparent) !important;
          box-shadow: 0 15px 36px rgba(0, 0, 0, .19);
        }
        :is(
          .st-key-product_data_card,
          .st-key-product_insight_card,
          .st-key-product_model_card,
          .st-key-pipeline_review_card,
          .st-key-pipeline_clean_card,
          .st-key-pipeline_vector_card,
          .st-key-pipeline_result_card
        ) [data-testid="stIconMaterial"] {
          color: var(--card-accent);
        }

        h1, h2, h3 {
          color: var(--it-text);
          letter-spacing: -.03em;
        }
        :is(h2, h3, h4) [data-testid="stIconMaterial"] {
          color: var(--it-blue);
        }
        h1 { font-size: clamp(1.8rem, 2.4vw, 2.35rem); }
        h2 { font-size: 1.55rem; }
        h3 { font-size: 1.22rem; }
        p { line-height: 1.55; }

        @media (max-width: 900px) {
          [data-testid="stMainBlockContainer"] {
            padding-top: 3.5rem;
            padding-inline: 1rem;
          }
          [data-testid="stMetric"] { min-height: 102px; }
        }
        @media (prefers-reduced-motion: reduce) {
          [data-testid="stSidebarNavLink"],
          [data-testid="stBaseButton-primary"],
          [data-testid="stBaseButton-secondary"],
          [data-testid="stButtonGroup"] button,
          [data-testid="stMetric"],
          [data-testid="stVerticalBlockBorderWrapper"],
          [class*="st-key-product_"],
          [class*="st-key-pipeline_"] { transition: none; }
        }
        </style>
        """
    )
