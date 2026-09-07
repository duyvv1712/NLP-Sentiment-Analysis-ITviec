"""Shared visual styling and the compact Streamlit theme picker."""

from __future__ import annotations

import streamlit as st


_THEME_PICKER = st.components.v2.component(
    "itviec_theme_picker",
    html="""
        <div class="theme-options" role="group" aria-label="Chọn giao diện">
          <button type="button" data-mode="System" title="Theo hệ thống" aria-label="Theo hệ thống">
            <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/></svg>
            <span>Hệ thống</span>
          </button>
          <button type="button" data-mode="Light" title="Giao diện sáng" aria-label="Giao diện sáng">
            <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.66 6.34l1.41-1.41"/></svg>
            <span>Sáng</span>
          </button>
          <button type="button" data-mode="Dark" title="Giao diện tối" aria-label="Giao diện tối">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 14.1A8.5 8.5 0 0 1 9.9 3.5 8.5 8.5 0 1 0 20.5 14.1Z"/></svg>
            <span>Tối</span>
          </button>
        </div>
    """,
    css="""
        .theme-options {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .55rem;
          padding: .15rem;
        }
        .theme-options button {
          appearance: none;
          border: 1px solid var(--st-border-color);
          border-radius: 14px;
          background: color-mix(in srgb, var(--st-secondary-background-color) 82%, transparent);
          color: var(--st-text-color);
          cursor: pointer;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: .38rem;
          min-height: 76px;
          padding: .65rem .25rem;
          font: inherit;
          font-size: .76rem;
          transition: transform 150ms ease, border-color 150ms ease, background 150ms ease;
        }
        .theme-options button:hover {
          transform: translateY(-2px);
          border-color: var(--st-primary-color);
          background: color-mix(in srgb, var(--st-primary-color) 12%, var(--st-secondary-background-color));
        }
        .theme-options button.is-active {
          border-color: var(--st-primary-color);
          background: color-mix(in srgb, var(--st-primary-color) 18%, var(--st-secondary-background-color));
          color: var(--st-primary-color);
          font-weight: 600;
          box-shadow: 0 0 0 1px color-mix(in srgb, var(--st-primary-color) 38%, transparent);
        }
        .theme-options svg {
          width: 24px;
          height: 24px;
          fill: none;
          stroke: currentColor;
          stroke-width: 1.8;
          stroke-linecap: round;
          stroke-linejoin: round;
        }
        @media (prefers-reduced-motion: reduce) {
          .theme-options button { transition: none; }
        }
    """,
    js="""
      export default function (component) {
        const { parentElement } = component;
        const storageKey = "itviec-theme-mode";
        const validModes = ["System", "Light", "Dark"];
        let selectedMode = localStorage.getItem(storageKey) || "Dark";
        if (!validModes.includes(selectedMode)) selectedMode = "Dark";

        const paintSelection = () => {
          parentElement.querySelectorAll("button[data-mode]").forEach((button) => {
            const active = button.dataset.mode === selectedMode;
            button.classList.toggle("is-active", active);
            button.setAttribute("aria-pressed", String(active));
          });
        };

        const selectNativeTheme = (mode) => {
          const menuButton = document.querySelector('[data-testid="stMainMenuButton"]');
          if (!menuButton) return;

          if (menuButton.getAttribute("aria-expanded") !== "true") menuButton.click();
          window.setTimeout(() => {
            const option = document.querySelector(`[data-testid="stMainMenuItem-theme-${mode}"]`);
            if (!option) return;
            option.click();
            selectedMode = mode;
            localStorage.setItem(storageKey, mode);
            paintSelection();
            window.setTimeout(() => {
              if (menuButton.getAttribute("aria-expanded") === "true") menuButton.click();
            }, 30);
          }, 60);
        };

        const buttons = parentElement.querySelectorAll("button[data-mode]");
        buttons.forEach((button) => {
          button.onclick = () => selectNativeTheme(button.dataset.mode);
        });
        paintSelection();

        return () => buttons.forEach((button) => { button.onclick = null; });
      }
    """,
)


def apply_app_style() -> None:
    """Add a restrained, theme-aware visual layer around native widgets."""
    st.html(
        """
        <style>
        /* Keep the sidebar control available while removing the top-right chrome. */
        header[data-testid="stHeader"] {
          background: transparent;
        }
        header[data-testid="stHeader"] [data-testid="stToolbar"] > div > div:last-child {
          display: none !important;
        }
        [data-testid="stMainMenuPopover"] {
          opacity: 0 !important;
          pointer-events: none !important;
        }

        [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(circle at 88% -4%, color-mix(in srgb, var(--st-primary-color) 17%, transparent) 0, transparent 32rem),
            radial-gradient(circle at 4% 92%, color-mix(in srgb, var(--st-blue-color) 8%, transparent) 0, transparent 27rem),
            var(--st-background-color);
        }
        section[data-testid="stSidebar"] {
          box-shadow: 10px 0 36px color-mix(in srgb, #020617 16%, transparent);
        }
        [data-testid="stSidebarNavLink"] {
          border-radius: 12px;
          margin-block: 2px;
          transition: background 150ms ease, transform 150ms ease;
        }
        [data-testid="stSidebarNavLink"]:hover {
          transform: translateX(3px);
        }
        [data-testid="stSidebarNavLink"][aria-current="page"] {
          background: linear-gradient(100deg,
            color-mix(in srgb, var(--st-primary-color) 24%, transparent),
            color-mix(in srgb, var(--st-primary-color) 7%, transparent));
          box-shadow: inset 3px 0 0 var(--st-primary-color);
        }

        /* Buttons and compact selectors share one clear, tactile visual language. */
        [data-testid="stBaseButton-primary"],
        [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
          min-height: 42px;
          padding-inline: 1.2rem;
          border: 0;
          background: linear-gradient(135deg, var(--st-primary-color), var(--st-blue-color));
          box-shadow: 0 8px 20px color-mix(in srgb, var(--st-primary-color) 28%, transparent);
          font-weight: 600;
          transition: transform 150ms ease, filter 150ms ease, box-shadow 150ms ease;
        }
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover {
          transform: translateY(-2px);
          filter: brightness(1.08);
          box-shadow: 0 11px 26px color-mix(in srgb, var(--st-primary-color) 36%, transparent);
        }
        [data-testid="stBaseButton-secondary"],
        [data-testid="stPopoverButton"] > button {
          min-height: 42px;
          border-color: color-mix(in srgb, var(--st-primary-color) 28%, var(--st-border-color));
          background: color-mix(in srgb, var(--st-secondary-background-color) 88%, transparent);
          font-weight: 500;
          transition: transform 150ms ease, border-color 150ms ease, background 150ms ease;
        }
        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stPopoverButton"] > button:hover {
          transform: translateY(-1px);
          border-color: var(--st-primary-color);
          background: color-mix(in srgb, var(--st-primary-color) 11%, var(--st-secondary-background-color));
        }

        [data-testid="stButtonGroup"] [role="toolbar"],
        [data-testid="stButtonGroup"] [role="radiogroup"] {
          gap: .38rem;
          padding: .3rem;
          border: 1px solid color-mix(in srgb, var(--st-border-color) 85%, transparent);
          border-radius: 14px;
          background: color-mix(in srgb, var(--st-secondary-background-color) 82%, transparent);
        }
        [data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
          min-height: 38px;
          margin: 0;
          border: 0;
          border-radius: 10px !important;
          transition: background 150ms ease, color 150ms ease, box-shadow 150ms ease;
        }
        [data-testid="stButtonGroup"] button[data-variant="segmented_control"][data-selected] {
          background: color-mix(in srgb, var(--st-primary-color) 20%, var(--st-secondary-background-color));
          color: var(--st-primary-color);
          box-shadow: 0 4px 12px color-mix(in srgb, var(--st-primary-color) 18%, transparent);
          font-weight: 600;
        }
        [data-testid="stButtonGroup"] button[data-variant="pills"] {
          min-height: 38px;
          padding-inline: 1rem;
          border-color: color-mix(in srgb, var(--st-primary-color) 24%, var(--st-border-color));
          transition: transform 150ms ease, background 150ms ease, border-color 150ms ease;
        }
        [data-testid="stButtonGroup"] button[data-variant="pills"]:hover {
          transform: translateY(-1px);
          border-color: var(--st-primary-color);
        }
        [data-testid="stButtonGroup"] button[data-variant="pills"][data-selected] {
          background: color-mix(in srgb, var(--st-primary-color) 18%, var(--st-secondary-background-color));
          color: var(--st-primary-color);
          border-color: var(--st-primary-color);
          font-weight: 600;
        }

        [data-testid="stTabs"] [role="tablist"] {
          gap: .4rem;
          padding: .35rem;
          border: 1px solid var(--st-border-color);
          border-radius: 14px;
          background: color-mix(in srgb, var(--st-secondary-background-color) 84%, transparent);
        }
        [data-testid="stTabs"] [role="tab"] {
          min-height: 40px;
          padding-inline: 1rem;
          border-radius: 10px;
          transition: background 150ms ease, color 150ms ease;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
          background: color-mix(in srgb, var(--st-primary-color) 18%, var(--st-secondary-background-color));
          color: var(--st-primary-color);
          font-weight: 600;
        }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"],
        [data-testid="stTabs"] [data-baseweb="tab-border"] {
          display: none;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stTextArea"] textarea {
          border-color: color-mix(in srgb, var(--st-primary-color) 24%, var(--st-border-color));
          background: color-mix(in srgb, var(--st-secondary-background-color) 72%, var(--st-background-color));
          transition: border-color 150ms ease, box-shadow 150ms ease;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
        [data-testid="stTextArea"] textarea:focus {
          border-color: var(--st-primary-color);
          box-shadow: 0 0 0 3px color-mix(in srgb, var(--st-primary-color) 18%, transparent);
        }
        [data-testid="stMetric"] {
          min-height: 126px;
          padding: 1rem 1.05rem;
          border: 1px solid color-mix(in srgb, var(--st-border-color) 82%, transparent);
          border-radius: 18px;
          background: linear-gradient(145deg,
            color-mix(in srgb, var(--st-secondary-background-color) 92%, transparent),
            color-mix(in srgb, var(--st-primary-color) 7%, var(--st-secondary-background-color)));
          box-shadow: 0 12px 30px color-mix(in srgb, #020617 10%, transparent);
          transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
        }
        [data-testid="stMetric"]:hover {
          transform: translateY(-3px);
          border-color: color-mix(in srgb, var(--st-primary-color) 58%, var(--st-border-color));
          box-shadow: 0 17px 38px color-mix(in srgb, #020617 15%, transparent);
        }
        [data-testid="stMetricValue"] {
          color: var(--st-primary-color);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
          box-shadow: 0 10px 28px color-mix(in srgb, #020617 8%, transparent);
          transition: border-color 160ms ease, box-shadow 160ms ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
          border-color: color-mix(in srgb, var(--st-primary-color) 45%, var(--st-border-color));
          box-shadow: 0 14px 34px color-mix(in srgb, #020617 12%, transparent);
        }
        h1, h2, h3 {
          letter-spacing: -.025em;
        }
        div[data-testid="stPopover"] > button {
          width: 100%;
          justify-content: flex-start;
        }
        @media (prefers-reduced-motion: reduce) {
          [data-testid="stSidebarNavLink"],
          [data-testid="stBaseButton-primary"],
          [data-testid="stBaseButton-secondary"],
          [data-testid="stButtonGroup"] button,
          [data-testid="stMetric"],
          [data-testid="stVerticalBlockBorderWrapper"] { transition: none; }
        }
        </style>
        """
    )


def render_theme_picker() -> None:
    """Render the single bottom-left control that reveals three theme icons."""
    with st.popover("Giao diện", icon=":material/palette:", width="stretch"):
        st.caption("Chọn chế độ hiển thị")
        _THEME_PICKER(key="itviec-theme-picker")
