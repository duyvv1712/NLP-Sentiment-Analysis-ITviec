from pathlib import Path
import tomllib

from streamlit.testing.v1 import AppTest

from src.app_services import get_model_status


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_theme_config_defines_system_light_and_dark_modes():
    config = tomllib.loads((PROJECT_ROOT / ".streamlit" / "config.toml").read_text())
    theme = config["theme"]

    assert theme["base"] == "dark"
    assert theme["light"]["backgroundColor"] == "#F8FAFF"
    assert theme["dark"]["backgroundColor"] == "#07111F"
    assert theme["dark"]["sidebar"]["backgroundColor"] == "#081525"
    assert theme["light"]["textColor"] != theme["dark"]["textColor"]
    assert theme["light"]["sidebar"]["backgroundColor"] != theme["dark"]["sidebar"]["backgroundColor"]


def test_theme_picker_uses_valid_component_v2_module():
    source = (PROJECT_ROOT / "src" / "app_theme.py").read_text()

    assert "export default function (component)" in source
    assert "const { parentElement } = component" in source
    assert "root.querySelectorAll" not in source


def test_streamlit_entrypoint_renders_default_page():
    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=90).run()

    assert not app.exception
    assert any("Hiểu tiếng nói" in title.value for title in app.title)
    assert "render_theme_picker()" in (PROJECT_ROOT / "app.py").read_text()


def test_prediction_page_handles_model_handoff_state():
    app = AppTest.from_file(
        PROJECT_ROOT / "app_pages" / "predict.py", default_timeout=90
    ).run()
    app.text_area[0].set_value("Môi trường tốt nhưng thường xuyên OT không lương")
    app.button[0].click().run()

    assert not app.exception
    if not get_model_status().ready:
        assert any("model chưa được bàn giao" in item.value for item in app.warning)
        assert app.code


def test_company_insights_page_renders_real_dataset():
    app = AppTest.from_file(
        PROJECT_ROOT / "app_pages" / "insights.py", default_timeout=90
    ).run()

    assert not app.exception
    assert any("Insight cảm xúc" in title.value for title in app.title)
    assert app.metric
    assert any(
        "WordCloud chỉ được tạo" in caption.value for caption in app.caption
    )
