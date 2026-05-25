"""Streamlit UI for the Intelligent Reading Comprehension and Quiz Generation System."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from html import escape
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluate import plot_confusion_matrix  # noqa: E402
from inference import RaceInferenceEngine  # noqa: E402
try:
    from scripts.ensure_kaggle_data_on_streamlit import ensure_dataset_available  # noqa: E402
except ModuleNotFoundError:
    ensure_dataset_available = None



st.set_page_config(layout="wide", page_title="RACE Quiz AI", page_icon="🧠")



def apply_theme() -> None:
    """Apply custom CSS for a more polished product-style interface."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        :root {
            --rc-bg: #0b1220;
            --rc-panel: rgba(17, 24, 39, 0.72);
            --rc-ink: #e6edf6;
            --rc-muted: #b6c2d6;
            --rc-line: rgba(148, 163, 184, 0.22);
            --rc-blue: #60a5fa;
            --rc-blue-hover: #93c5fd;
            --rc-green: #34d399;
            --rc-red: #fb7185;
            --rc-amber: #fbbf24;
            --rc-text-strong: #ffffff;
        }

        /* Better default contrast for Streamlit text */
        .stMarkdown, .stText, .stCode, .stTable, .stDataFrame {
            color: var(--rc-ink) !important;
        }

        .stMarkdown * {
            color: inherit !important;
        }
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
            color: var(--rc-ink) !important;
        }
        
        .stApp {
            background-color: var(--rc-bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1280px;
        }
        
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.6) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-right: 1px solid var(--rc-line);
        }
        
        div[data-testid="stButton"] button {
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.05);
            font-weight: 600;
            min-height: 3rem;
            transition: all 0.25s ease;
            color: white !important;
        }
        div[data-testid="stButton"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            border-color: rgba(255, 255, 255, 0.4);
            background: rgba(255, 255, 255, 0.1);
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            border: none;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background: linear-gradient(135deg, #60a5fa, #3b82f6);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        }

        .rc-hero {
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--rc-line);
            border-radius: 16px;
            padding: 40px 48px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        }
        .rc-hero::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #10b981, #f59e0b);
        }
        .rc-hero h1 {
            font-size: 2.8rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0 0 12px 0;
            background: linear-gradient(to right, #ffffff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .rc-hero p {
            color: #cbd5e1;
            margin: 0;
            max-width: 800px;
            font-size: 1.1rem;
            font-weight: 300;
        }

        .rc-card {
            background: var(--rc-panel);
            backdrop-filter: blur(10px);
            border: 1px solid var(--rc-line);
            border-radius: 12px;
            padding: 24px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .rc-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.3);
            border-color: rgba(255, 255, 255, 0.2);
        }
        .rc-card h3 {
            margin: 0 0 8px 0;
            font-size: 1.15rem;
            font-weight: 600;
            color: #f8fafc;
        }
        .rc-muted {
            color: var(--rc-muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .rc-question {
            background: rgba(59, 130, 246, 0.1);
            border-left: 4px solid #3b82f6;
            padding: 20px 24px;
            border-radius: 0 12px 12px 0;
            color: #f8fafc;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .rc-option {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
            background: rgba(30, 41, 59, 0.4);
            transition: all 0.2s ease;
            font-size: 1.05rem;
        }
        .rc-option:hover {
            background: rgba(59, 130, 246, 0.1);
            border-color: rgba(59, 130, 246, 0.4);
        }

        .rc-kpi {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid var(--rc-line);
            border-radius: 12px;
            padding: 16px 20px;
            backdrop-filter: blur(8px);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .rc-kpi .label {
            color: var(--rc-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }
        .rc-kpi .value {
            color: #f8fafc;
            font-size: 1.75rem;
            font-weight: 700;
            margin-top: 6px;
        }

        .rc-pill {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.1);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.2);
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .rc-warning {
            border: 1px solid rgba(245, 158, 11, 0.3);
            background: rgba(245, 158, 11, 0.1);
            color: #fcd34d;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            font-weight: 500;
        }

        textarea, input, .stSelectbox > div > div {
            background: rgba(3, 7, 18, 0.55) !important;
            border: 1px solid rgba(148, 163, 184, 0.28) !important;
            border-radius: 12px !important;
            color: var(--rc-ink) !important;
        }

        textarea::placeholder, input::placeholder {
            color: rgba(182, 194, 214, 0.75) !important;
        }

        div[role="textbox"], .stMarkdown {
            color: var(--rc-ink) !important;
        }
        textarea:focus, input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background: rgba(30, 41, 59, 0.4);
            padding: 8px 16px;
            border-radius: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px;
            color: var(--rc-muted);
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            color: white !important;
            border-bottom: 2px solid #3b82f6 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="rc-hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="rc-kpi"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="rc-card"><h3>{escape(title)}</h3><div class="rc-muted">{escape(body)}</div></div>',
        unsafe_allow_html=True,
    )


def warning_panel(message: str) -> None:
    st.markdown(f'<div class="rc-warning">{escape(message)}</div>', unsafe_allow_html=True)


@st.cache_resource
def load_engine() -> RaceInferenceEngine:
    return RaceInferenceEngine(model_dir=os.path.join(ROOT_DIR, "models"))


def init_state() -> None:
    defaults = {
        "screen": "📄 Article Input",
        "article_text": "",
        "quiz_result": None,
        "attempts": [],
        "hints_revealed": 1,
        "answer_checked": False,
        "selected_option": None,
        "last_latency_ms": 0.0,
        "race_question": None,
        "race_answer": None,
        "race_options": None,
        "sample_error": None,
        "sample_success": None,
        "load_sample_clicked": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_data
def _load_test_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_random_sample() -> tuple[str | None, str | None]:
    """Load one random RACE CSV article into session state.

    Streamlit deployments may not include all RACE splits.
    We try test.csv first, then fall back to val.csv and train.csv.
    """

    candidate_files = [
        ("test.csv", os.path.join(ROOT_DIR, "data", "raw", "test.csv")),
        ("val.csv", os.path.join(ROOT_DIR, "data", "raw", "val.csv")),
        ("train.csv", os.path.join(ROOT_DIR, "data", "raw", "train.csv")),
    ]

    for name, path in candidate_files:
        if not os.path.exists(path):
            continue

        try:
            df = _load_test_csv(path)
            if df.empty:
                return None, f"❌ {name} is empty. No articles available to load."

            sample = df.sample(
                1,
                random_state=np.random.default_rng().integers(0, 1_000_000),
            ).iloc[0]

            article = str(sample.get("article", ""))
            if pd.isna(article) or article.strip() == "":
                return None, "❌ Selected article is empty. Try loading another sample."

            answer_label = str(sample.get("answer", "")).strip().upper()
            st.session_state["race_question"] = str(sample.get("question", ""))
            st.session_state["race_answer"] = str(sample.get(answer_label, ""))
            st.session_state["race_options"] = {
                lbl: str(sample.get(lbl, "")) for lbl in ["A", "B", "C", "D"]
            }

            return article, None

        except pd.errors.ParserError as e:
            st.session_state.pop("race_question", None)
            st.session_state.pop("race_answer", None)
            st.session_state.pop("race_options", None)
            return None, f"❌ CSV parsing error in {name}: {str(e)[:120]}..."
        except MemoryError:
            st.session_state.pop("race_question", None)
            st.session_state.pop("race_answer", None)
            st.session_state.pop("race_options", None)
            return None, f"❌ {name} too large to load in memory."
        except Exception as e:
            st.session_state.pop("race_question", None)
            st.session_state.pop("race_answer", None)
            st.session_state.pop("race_options", None)
            return None, f"❌ Unexpected error in {name}: {str(e)}"

    missing = [name for name, path in candidate_files if not os.path.exists(path)]
    return None, (
        "❌ No RACE CSV splits found in data/raw. "
        "Expected test.csv / val.csv / train.csv. "
        f"Missing: {', '.join(missing)}"
    )


def sidebar_nav() -> None:
    screens = ["📄 Article Input", "❓ Quiz", "💡 Hints", "📊 Analytics"]
    st.sidebar.markdown("<h2 style='margin-bottom: 0;'>🧠 RACE Quiz AI</h2>", unsafe_allow_html=True)
    st.sidebar.caption("Traditional ML reading comprehension lab")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    current = st.session_state.get("screen", screens[0])
    selected = st.sidebar.radio(
        "Navigation",
        screens,
        index=screens.index(current) if current in screens else 0,
        label_visibility="collapsed",
    )
    st.session_state["screen"] = selected
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    if st.sidebar.button("Reset Session"):
        for key in [
            "quiz_result",
            "attempts",
            "answer_checked",
            "selected_option",
            "race_question",
            "race_answer",
            "race_options",
        ]:
            if key == "attempts":
                st.session_state[key] = []
            else:
                st.session_state[key] = None
        st.session_state["screen"] = "📄 Article Input"


def home_screen(engine: RaceInferenceEngine) -> None:
    # Ensure RACE CSV splits exist in deployed filesystem.
    # If download isn't possible (no network / no Kaggle token), app falls back to demo behavior.
    if ensure_dataset_available is not None:
        try:
            ensure_dataset_available(force=False)
        except Exception:
            # Do not block app rendering.
            pass

    hero(
        "RACE Quiz AI",
        "Generate reading-comprehension questions, distractors, hints, and model analytics using traditional machine learning.",
    )
    if engine.use_demo_mode:
        warning_panel(
            "Demo mode is active because one or more trained model files were not found. Rule-based fallbacks are being used."
        )

    if st.session_state.get("load_sample_clicked"):
        st.session_state["load_sample_clicked"] = False
        loaded_article, error_msg = load_random_sample()
        if error_msg:
            st.session_state["sample_error"] = error_msg
            st.session_state["sample_success"] = None
        else:
            st.session_state["article_text"] = loaded_article
            word_count = len(str(loaded_article).split())
            st.session_state["sample_success"] = (
                f"Loaded random sample ({word_count} words, {len(loaded_article)} characters)"
            )
            st.session_state["sample_error"] = None

    c1, c2, c3 = st.columns(3)
    with c1:
        info_card("Model A", "Answer verification with sparse OHE features, lexical overlap, and voting ensembles.")
    with c2:
        info_card("Model B", "Distractor ranking, diversity checks, and graduated hint generation.")
    with c3:
        info_card("Evaluation", "Session logs, confusion matrix, metric cards, and exportable attempts.")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([2.5, 1], gap="large")
    with left:
        st.markdown("<h3 style='margin-bottom: 12px;'>Reading Passage</h3>", unsafe_allow_html=True)
        article = st.text_area(
            "Paste a reading passage",
            key="article_text",
            height=320,
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate Quiz", type="primary", use_container_width=True):
            if not article.strip():
                st.warning("Paste or load an article before generating a quiz.")
                return
            try:
                with st.spinner("Generating quiz..."):
                    race_q = st.session_state.get("race_question")
                    race_a = st.session_state.get("race_answer")
                    st.session_state["quiz_result"] = engine.run_full_pipeline(
                        article,
                        question=race_q,
                        answer=race_a,
                    )
                    st.session_state["last_latency_ms"] = st.session_state["quiz_result"].get("inference_time_ms", 0.0)
                    st.session_state["hints_revealed"] = 1
                    st.session_state["answer_checked"] = False
                    st.session_state["selected_option"] = None
                    st.session_state["race_question"] = None
                    st.session_state["race_answer"] = None
                    st.session_state["screen"] = "❓ Quiz"
                st.rerun()
            except Exception as exc:
                st.error(f"Quiz generation failed: {exc}")

    with right:
        st.markdown("<h3 style='margin-bottom: 12px;'>Passage Tools</h3>", unsafe_allow_html=True)
        kpi_card("Words", str(len(article.split()) if article else 0))
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
        kpi_card("Characters", str(len(article) if article else 0))
        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        st.button(
            "Load Random RACE Sample",
            on_click=lambda: st.session_state.__setitem__("load_sample_clicked", True),
            use_container_width=True,
        )


def quiz_screen(engine: RaceInferenceEngine) -> None:
    result = st.session_state.get("quiz_result")
    if not result:
        st.info("Generate a quiz from the Article Input screen first.")
        return

    hero("Quiz Workspace", "Read the passage, select an answer, then inspect hints and model confidence.")
    with st.expander("Read the Passage", expanded=False):
        st.write(st.session_state.get("article_text", ""))

    st.markdown(f'<div class="rc-question">{escape(result["question"])}</div>', unsafe_allow_html=True)
    st.markdown("#### Options")
    for label, value in result["options"].items():
        st.markdown(
            f'<div class="rc-option"><b>{escape(label)}.</b> {escape(str(value))}</div>',
            unsafe_allow_html=True,
        )

    labels = list(result["options"].keys())
    selected = st.radio(
        "Choose an answer",
        labels,
        format_func=lambda label: f"{label}. {result['options'][label]}",
        key="selected_option",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        if st.button("Check My Answer", type="primary", use_container_width=True):
            correct = selected == result["correct"]
            row = {
                "article_snippet": st.session_state.get("article_text", "")[:120],
                "question": result["question"],
                "user_answer": selected,
                "correct": correct,
                "latency_ms": result.get("inference_time_ms", 0.0),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            st.session_state["attempts"].append(row)
            st.session_state["answer_checked"] = True

    with col2:
        if st.button("Get Hints", width="stretch"):
            st.session_state["screen"] = "💡 Hints"
            st.rerun()

    with col3:
        if st.button("Try Another Question", width="stretch"):
            st.session_state["screen"] = "📄 Article Input"
            st.session_state["quiz_result"] = None
            st.rerun()

    if st.session_state.get("answer_checked"):
        if selected == result["correct"]:
            st.success(
                f"Correct. The answer is {result['correct']}: {result['options'][result['correct']]}."
            )
        else:
            hint = result.get("hints", ["Review the passage."])[0]
            st.error(
                f"Not quite. Correct answer: {result['correct']}. {result['options'][result['correct']]}. Hint: {hint}"
            )

    st.markdown("<br><br>", unsafe_allow_html=True)
    metric_cols = st.columns(3, gap="large")
    with metric_cols[0]:
        kpi_card("Model Confidence", f"{float(result.get('confidence', 0.0)):.2f}")
    with metric_cols[1]:
        kpi_card("Latency", f"{result.get('inference_time_ms', 0)} ms")
    with metric_cols[2]:
        kpi_card("Mode", "Demo" if result.get("demo_mode") else "Trained")
    st.progress(max(0.0, min(1.0, float(result.get("confidence", 0.0)))))


def hints_screen() -> None:
    result = st.session_state.get("quiz_result")
    if not result:
        st.info("Generate a quiz first.")
        return

    hero("Hint Panel", "Reveal support gradually before showing the final answer.")
    with st.expander("Passage", expanded=False):
        st.write(st.session_state.get("article_text", ""))

    hints = result.get(
        "hints",
        [
            "Review the main idea.",
            "Find the sentence related to the question.",
            "Look near the answer phrase.",
        ],
    )
    labels = [
        "💡 Hint 1 — General Clue",
        "💡 Hint 2 — More Specific",
        "💡 Hint 3 — Near Explicit",
    ]

    revealed = st.session_state.get("hints_revealed", 1)
    for idx, label in enumerate(labels, start=1):
        if revealed >= idx:
            with st.expander(label, expanded=idx == revealed):
                st.write(hints[idx - 1] if idx - 1 < len(hints) else "Re-read the most relevant sentence.")
                if idx < 3 and st.button(f"Reveal Hint {idx + 1}", key=f"reveal_{idx}"):
                    st.session_state["hints_revealed"] = idx + 1
                    st.rerun()
        else:
            st.markdown(f'<span class="rc-pill">{escape(label)} locked</span>', unsafe_allow_html=True)

    if st.session_state.get("hints_revealed", 1) >= 3:
        if st.button("Reveal Answer", type="primary"):
            st.success(
                f"Answer: {result['correct']}. {result['options'][result['correct']]}")

    if st.button("Back to Quiz"):
        st.session_state["screen"] = "❓ Quiz"
        st.rerun()


def _load_pickle(path: str, default: Any) -> Any:
    try:
        return joblib.load(path)
    except Exception:
        return default


def _file_size(path: str) -> str:
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    except OSError:
        return "missing"


def analytics_screen(engine: RaceInferenceEngine) -> None:
    hero("Analytics Dashboard", "Inspect model behavior, session outcomes, artifacts, and runtime details.")
    tabs = st.tabs(["Model A Performance", "Model B Performance", "Session Log", "System Info"])

    model_a_results = _load_pickle(
        os.path.join(ROOT_DIR, "models", "model_a", "traditional", "results.pkl"), {}
    )
    model_b_results = _load_pickle(
        os.path.join(ROOT_DIR, "models", "model_b", "traditional", "results.pkl"), {}
    )

    with tabs[0]:
        metrics_df = model_a_results.get("metrics") if isinstance(model_a_results, dict) else None
        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            best = metrics_df.sort_values("macro_f1", ascending=False).iloc[0]
            st.markdown("#### Top Model (by Macro F1)")
            cols = st.columns(5)

            def _fmt_kpi(val: Any) -> str:
                if pd.isna(val):
                    return "N/A"
                return f"{val:.3f}"

            with cols[0]:
                kpi_card("Accuracy", _fmt_kpi(best.get("accuracy", 0)))
            with cols[1]:
                kpi_card("Macro F1", _fmt_kpi(best.get("macro_f1", 0)))
            with cols[2]:
                kpi_card("ROC AUC", _fmt_kpi(best.get("roc_auc", 0)))
            with cols[3]:
                kpi_card("PR AUC", _fmt_kpi(best.get("pr_auc", 0)))
            with cols[4]:
                kpi_card("Brier Score", _fmt_kpi(best.get("brier_score", 0)))

            st.markdown("<br>", unsafe_allow_html=True)
            cm = model_a_results.get("confusion_matrix", np.asarray([[0, 0], [0, 0]]))
            st.pyplot(plot_confusion_matrix(cm, ["Wrong", "Correct"], "Model A Confusion Matrix"))

            chart_df = metrics_df[
                metrics_df["model"].astype(str).str.contains(
                    "Logistic|SVC|NB|Ensemble", case=False, regex=True
                )
            ]
            st.plotly_chart(
                px.bar(
                    chart_df,
                    x="model",
                    y=["macro_f1", "roc_auc"],
                    barmode="group",
                    title="Model Comparisons",
                ),
                use_container_width=True,
            )

        else:
            st.warning("Model A results not found. Train Model A to populate this dashboard.")

        latencies = [
            a.get("latency_ms", 0)
            for a in st.session_state.get("attempts", [])
            if "latency_ms" in a
        ]
        latency_str = f"{np.mean(latencies):.1f} ms" if latencies else "No data yet"
        st.metric("Average inference latency", latency_str)

    with tabs[1]:
        if isinstance(model_b_results, dict) and model_b_results:
            cols = st.columns(5)
            with cols[0]:
                kpi_card("Ranker Acc", f"{model_b_results.get('distractor_ranker_accuracy', 0):.3f}")
            with cols[1]:
                kpi_card("Dist F1", f"{model_b_results.get('distractor_f1', 0):.3f}")
            with cols[2]:
                kpi_card("Dist Jaccard", f"{model_b_results.get('distractor_jaccard', 0):.3f}")
            with cols[3]:
                kpi_card("Hit Rate@3", f"{model_b_results.get('distractor_hit_rate_at_3', 0):.3f}")
            with cols[4]:
                kpi_card("Diversity", f"{model_b_results.get('pairwise_cosine_diversity', 0):.3f}")

            st.markdown("#### Distractor Generation (NLP)")
            nlp_cols = st.columns(3)
            with nlp_cols[0]:
                kpi_card("BLEU", f"{model_b_results.get('distractor_bleu', 0):.3f}")
            with nlp_cols[1]:
                kpi_card("ROUGE-L", f"{model_b_results.get('distractor_rougeL', 0):.3f}")
            with nlp_cols[2]:
                kpi_card("METEOR", f"{model_b_results.get('distractor_meteor', 0):.3f}")

            st.markdown("<br>", unsafe_allow_html=True)
            hint_df = pd.DataFrame(
                {
                    "K": ["P@1", "P@2", "P@3"],
                    "Precision": [
                        0.0,
                        0.0,
                        model_b_results.get("hint_precision_at_k", 0),
                    ],
                }
            )
            st.plotly_chart(
                px.bar(hint_df, x="K", y="Precision", title="Hint Precision@K"),
                use_container_width=True,
            )
        else:
            st.warning("Model B results not found. Train Model B to populate metrics.")

        sample = pd.DataFrame(
            [
                {"example": 1, "distractor": "school library", "quality_note": "plausible location"},
                {"example": 2, "distractor": "next morning", "quality_note": "time-based alternative"},
                {"example": 3, "distractor": "different reason", "quality_note": "semantic foil"},
                {"example": 4, "distractor": "classmate", "quality_note": "person-based alternative"},
                {"example": 5, "distractor": "main problem", "quality_note": "topic-level foil"},
            ]
        )
        st.dataframe(sample, use_container_width=True)

    with tabs[2]:
        attempts = pd.DataFrame(st.session_state.get("attempts", []))
        st.dataframe(attempts, use_container_width=True)
        total = len(attempts)
        correct = int(attempts["correct"].sum()) if total and "correct" in attempts else 0
        st.metric("Total attempts", total)
        st.metric("Correct", correct)
        st.metric("Session accuracy", f"{(correct / total * 100) if total else 0:.1f}%")
        st.download_button(
            "Export to CSV",
            attempts.to_csv(index=False),
            "session_log.csv",
            "text/csv",
            disabled=attempts.empty,
        )

    with tabs[3]:
        model_files = [
            os.path.join(ROOT_DIR, "models", "model_a", "traditional", "logistic_regression.pkl"),
            os.path.join(ROOT_DIR, "models", "model_a", "traditional", "linear_svc_calibrated.pkl"),
            os.path.join(ROOT_DIR, "models", "model_b", "traditional", "model_b_artifacts.pkl"),
        ]
        st.dataframe(
            pd.DataFrame(
                {
                    "file": [os.path.basename(p) for p in model_files],
                    "size": [_file_size(p) for p in model_files],
                }
            )
        )
        vocab_size = (
            len(getattr(engine.ohe_vectorizer, "vocabulary_", {}))
            if engine.ohe_vectorizer is not None
            else 0
        )
        st.write(f"Vocabulary size: {vocab_size}")
        st.write(f"Python: {platform.python_version()}")
        st.write(f"Streamlit: {st.__version__}")

        try:
            import sklearn

            st.write(f"scikit-learn: {sklearn.__version__}")
        except Exception:
            st.write("scikit-learn: unavailable")

        try:
            import torch

            st.write(f"GPU available: {torch.cuda.is_available()}")
        except Exception:
            st.write("GPU available: torch not installed; traditional ML does not require GPU.")


def main() -> None:
    apply_theme()
    init_state()
    engine = load_engine()
    sidebar_nav()

    screen = st.session_state.get("screen", "📄 Article Input")
    if screen == "📄 Article Input":
        home_screen(engine)
    elif screen == "❓ Quiz":
        quiz_screen(engine)
    elif screen == "💡 Hints":
        hints_screen()
    elif screen == "📊 Analytics":
        analytics_screen(engine)


if __name__ == "__main__":
    main()

