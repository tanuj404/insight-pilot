import streamlit as st
import numpy as np
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
from industry_knowledge import INDUSTRY_KNOWLEDGE
import html
import io
import re
import textwrap

import charts
import data_utils
import guardrails
import llm_client
import pipeline

# ── API key handling: works locally AND when deployed ──
load_dotenv()

api_key = None
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

# Page setup — must come before any other st.* calls
st.set_page_config(page_title="InsightPilot", layout="wide")

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS — Analyst Pro polish
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
    }
    h2, h3 {
        font-weight: 600;
        color: #1E293B;
    }
    .stAlert {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .stButton > button {
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #2563EB;
        color: #2563EB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: #64748B;
        padding: 8px 4px;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .stDataFrame {
        border: 1px solid #E2E8F0;
        border-radius: 6px;
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #CBD5E1;
        border-radius: 8px;
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# LOGO — line-art bar-chart mark, used in hero + sidebar
# ══════════════════════════════════════════════════════════════════
def logo_svg(size=56):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block;">
        <circle cx="32" cy="32" r="29" stroke="#334155" stroke-width="2"/>
        <rect x="19" y="34" width="5" height="12" rx="1" fill="#334155"/>
        <rect x="27" y="27" width="5" height="19" rx="1" fill="#334155"/>
        <rect x="35" y="20" width="5" height="26" rx="1" fill="#334155"/>
        <path d="M19 29 L28 22 L37 17 L44 13" stroke="#334155" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        <path d="M38 13 L44 13 L44 19" stroke="#334155" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    </svg>"""

# ── If no API key, ask the user ──
if not api_key:
    st.markdown(f"""
    <div style='text-align: center; padding: 24px 0 8px 0;'>
        <div style='display: flex; justify-content: center;'>{logo_svg(56)}</div>
        <h1 style='margin: 12px 0 4px 0; padding: 0; font-size: 2.5rem;'>InsightPilot</h1>
    </div>
    """, unsafe_allow_html=True)
    st.info(
        "To use this app, enter your free Google Gemini API key below. "
        "Get one at https://aistudio.google.com/apikey"
    )
    api_key = st.text_input("Gemini API Key:", type="password")
    if not api_key:
        st.stop()

genai.configure(api_key=api_key)

QUOTA_MESSAGE = (
    "The AI service is busy right now (many people are using InsightPilot at once). "
    "Please wait about 30 seconds and try again."
)


@st.cache_data(ttl=3600, max_entries=1000, show_spinner=False)
def cached_llm(prompt: str) -> str:
    """Identical prompts (e.g. the same sample dataset + question) are answered once and shared.
    Errors are never cached, so a rate-limited call can simply be retried."""
    return llm_client.generate_text(prompt)


@st.cache_data(show_spinner=False)
def load_path(path: str, mtime: float):
    raw = data_utils.read_csv_robust(path)
    return data_utils.prepare_dataframe(raw)


@st.cache_data(show_spinner=False)
def load_upload(data: bytes, name: str):
    raw = data_utils.read_csv_robust(io.BytesIO(data))
    return data_utils.prepare_dataframe(raw)


RESET_KEYS = ["active_question", "last_answered_question", "last_answer",
              "kpi_result", "formula_result", "answer_id"]


def select_sample(path, name):
    """Load a bundled dataset into the app and reset any previous analysis."""
    st.session_state.sample_file = path
    st.session_state.sample_name = name
    for k in RESET_KEYS:
        st.session_state.pop(k, None)


# ══════════════════════════════════════════════════════════════════
# CENTERED HERO
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='text-align: center; padding: 24px 0 8px 0;'>
    <div style='display: flex; justify-content: center;'>{logo_svg(64)}</div>
    <h1 style='margin: 12px 0 4px 0; padding: 0; font-size: 2.5rem;'>InsightPilot</h1>
    <p style='color: #64748B; font-size: 1.05rem; margin: 0;'>
        AI-powered KPI advisor and analytics agent.<br>
        Upload a CSV to begin.
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR: Brand header (always visible)
# ══════════════════════════════════════════════════════════════════
st.sidebar.markdown(f"""
<div style='display: flex; align-items: center; gap: 10px; padding: 0 0 20px 0;'>
    <div>{logo_svg(32)}</div>
    <div>
        <div style='font-weight: 700; font-size: 1.2rem; color: #0F172A; line-height: 1.1;'>InsightPilot</div>
        <div style='font-size: 0.72rem; color: #94A3B8; margin-top: 2px;'>Upload • Industry • Insights</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR: Sample datasets + Clear button (always visible)
# ══════════════════════════════════════════════════════════════════
st.sidebar.header("📁 Try a sample")
st.sidebar.caption("Don't have a CSV? Load one of these to explore:")

sample_col1, sample_col2 = st.sidebar.columns(2)
with sample_col1:
    if st.button("🛒 Retail", width="stretch", key="sample_retail"):
        select_sample("sample_data/retail_sales.csv", "retail_sales.csv")
with sample_col2:
    if st.button("🏦 Banking", width="stretch", key="sample_banking"):
        select_sample("sample_data/banking_loans.csv", "banking_loans.csv")

st.sidebar.markdown("---")

# Clear / Start Over button
if st.sidebar.button("🔄 Clear / Start Over", width="stretch"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")

# ══════════════════════════════════════════════════════════════════
# MAIN: File uploader + data source resolution
# ══════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

df = None
data_source_name = None
prep_notes = []

try:
    if uploaded_file is not None:
        df, prep_notes = load_upload(uploaded_file.getvalue(), uploaded_file.name)
        data_source_name = uploaded_file.name
        st.session_state.pop("sample_file", None)
    elif "sample_file" in st.session_state:
        sample_path = st.session_state.sample_file
        df, prep_notes = load_path(sample_path, os.path.getmtime(sample_path))
        data_source_name = st.session_state.sample_name
except Exception as e:
    st.error(
        "That file could not be read as a CSV. Please re-export it as **CSV (UTF-8)** and try again."
    )
    with st.expander("Technical details"):
        st.code(f"{type(e).__name__}: {e}")
    df = None
    data_source_name = None

# ══════════════════════════════════════════════════════════════════
# SIDEBAR: Dataset Info + Schema (always visible, empty state until loaded)
# ══════════════════════════════════════════════════════════════════
st.sidebar.subheader("📄 Dataset Info")
if df is not None:
    st.sidebar.write(f"**{data_source_name}**")
    st.sidebar.caption(f"{df.shape[0]} rows · {df.shape[1]} columns")
else:
    st.sidebar.caption("200MB per file • dataset")

st.sidebar.markdown("---")
st.sidebar.subheader("Schema")
if df is not None:
    schema_df = pd.DataFrame({
        "Name": df.columns,
        "Data type": [data_utils.dtype_label(df[c]) for c in df.columns],
    })
else:
    schema_df = pd.DataFrame({"Name": [], "Data type": []})
st.sidebar.dataframe(schema_df, hide_index=True, width="stretch")
if df is not None and prep_notes:
    with st.sidebar.expander("Automatic data preparation"):
        for note in prep_notes:
            st.write(f"• {note}")

# ══════════════════════════════════════════════════════════════════
# THREE FEATURE CARDS — landing only
# ══════════════════════════════════════════════════════════════════
if df is None:
    st.markdown("""
<div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;
            margin-top: 28px; margin-bottom: 12px;'>

<div style='background: #1E293B; color: white; border-radius: 12px;
            padding: 28px 20px; text-align: center;'>
    <div style='font-size: 32px; margin-bottom: 12px;'>📁</div>
    <div style='font-weight: 600; font-size: 1.05rem; margin-bottom: 6px;'>
        Feed the Pilot
    </div>
    <div style='color: #94A3B8; font-size: 0.85rem; line-height: 1.5;'>
        Upload your raw business CSV data.
    </div>
</div>

<div style='background: #ECFDF5; color: #064E3B; border-radius: 12px;
            padding: 28px 20px; text-align: center; border: 1px solid #D1FAE5;'>
    <div style='font-size: 32px; margin-bottom: 12px;'>📊</div>
    <div style='font-weight: 600; font-size: 1.05rem; margin-bottom: 6px;'>
        Get KPI Briefing
    </div>
    <div style='color: #047857; font-size: 0.85rem; line-height: 1.5;'>
        Industry-specific DAX & Tableau formulas instantly.
    </div>
</div>

<div style='background: #EFF6FF; color: #1E3A8A; border-radius: 12px;
            padding: 28px 20px; text-align: center; border: 1px solid #DBEAFE;'>
    <div style='font-size: 32px; margin-bottom: 12px;'>🤖</div>
    <div style='font-weight: 600; font-size: 1.05rem; margin-bottom: 6px;'>
        Run AI Analytics
    </div>
    <div style='color: #1D4ED8; font-size: 0.85rem; line-height: 1.5;'>
        Click stakeholder questions to generate charts & insights.
    </div>
</div>

</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN WORKSPACE — appears once data is loaded
# ══════════════════════════════════════════════════════════════════
if df is not None:
    # ── Sidebar: data preview ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("Preview (first 5)")
    st.sidebar.dataframe(df.head(5), hide_index=True, width="stretch")

    # ── Auto-generate business questions when a new file/sample is loaded ──
    if st.session_state.get("last_uploaded_file") != data_source_name:
        with st.spinner("Reading your data and preparing insights..."):
            q_prompt = f"""You are a senior business analyst.

A dataset has these columns and values:
{pipeline.column_info(df)}

Generate exactly 5 specific business questions a manager or stakeholder
would realistically ask about THIS data. Make them concrete and answerable
from the columns above (e.g. "Which region has the highest revenue?").
Include at least one question about a trend over time if there is a date
column, and one ranking question.

Return ONLY the 5 questions, one per line, no numbering, no extra text."""

            questions_failed = False
            try:
                q_text = cached_llm(q_prompt)
                questions = []
                for line in q_text.split("\n"):
                    line = re.sub(r"^[\-\*•\d\.\)\s]+", "", line).strip()
                    if len(line) > 10:
                        questions.append(line)
                questions = questions[:5]
            except llm_client.QuotaExceeded:
                questions, questions_failed = [], True

            st.session_state.questions = questions
            st.session_state.questions_failed = questions_failed
            st.session_state.last_uploaded_file = data_source_name
            for k in RESET_KEYS:
                st.session_state.pop(k, None)

    # ══════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════
    tab_kpi, tab_qa = st.tabs(["🎯 KPI Briefing", "💬 Ask Questions"])

    # ──────────────────────────────────────────────────────────────────
    # TAB 1 — KPI Briefing
    # ──────────────────────────────────────────────────────────────────
    with tab_kpi:
        st.subheader("Select Your Industry")
        st.write("Different industries track different KPIs. Pick yours so the AI gives relevant suggestions.")

        industries = [
            "Generic (no specific industry)",
            "Banking & Finance",
            "Healthcare",
            "Retail & E-commerce",
            "Supply Chain & Logistics",
            "Manufacturing",
            "Insurance",
            "Telecommunications",
            "Education",
            "Government & Public Sector",
            "Real Estate",
            "Transportation",
            "Energy & Utilities",
            "Hospitality & Tourism",
            "Marketing & Advertising",
            "Human Resources",
            "Sports Analytics",
            "Pharmaceuticals",
            "Technology & SaaS",
            "Agriculture",
            "Non-Profit Organizations",
        ]

        selected_industry = st.selectbox("Choose your industry:", industries)
        st.info(f"You selected: **{selected_industry}**")

        if st.button("Generate KPI Suggestions"):
            with st.spinner("Analyzing your data..."):
                columns = list(df.columns)
                sample_rows = df.head(3).to_string()

                industry_info = INDUSTRY_KNOWLEDGE.get(selected_industry)
                if industry_info:
                    known_kpis = ", ".join(industry_info["kpis"])
                    problems = industry_info["problems"]
                else:
                    known_kpis = "general business KPIs"
                    problems = "general business analysis"

                prompt = f"""You are a senior data analyst.

A user uploaded a dataset.
Columns: {columns}
Here are 3 sample rows so you understand the data:
{sample_rows}

Their industry is: {selected_industry}
Typical problems in this industry: {problems}
Common KPIs in this industry: {known_kpis}

Use the industry KPIs above as a STARTING POINT, but also use your own
analysis of the actual columns and sample data to suggest additional
KPIs the user may not have thought of.

Recommend 5-8 KPIs they can realistically calculate from these columns.
For each KPI give:
- The KPI name (as a bold heading)
- One sentence on why it matters for their data

Only suggest KPIs that are actually possible given their columns."""

                try:
                    kpi_raw = cached_llm(prompt)
                except llm_client.QuotaExceeded:
                    kpi_raw = None
                if kpi_raw is None:
                    st.warning(QUOTA_MESSAGE)
                else:
                    # Strip code fences and unescape HTML so returned markup renders
                    kpi_text = kpi_raw.replace("```html", "").replace("```", "")
                    kpi_text = html.unescape(kpi_text)
                    # Remove common leading indentation and any leading newlines/spaces
                    kpi_text = textwrap.dedent(kpi_text).lstrip()
                    st.session_state.kpi_result = kpi_text

        if "kpi_result" in st.session_state:
            st.subheader("AI-Recommended KPIs")
            st.markdown(st.session_state.kpi_result, unsafe_allow_html=True)

            st.markdown("---")
            st.write("Want the formulas for these KPIs?")

            if st.button("Yes, generate DAX & Tableau formulas"):
                with st.spinner("Writing formulas..."):
                    formula_prompt = f"""You are a Power BI and Tableau expert.

Here are the KPIs recommended for a dataset with these columns: {list(df.columns)}

{st.session_state.kpi_result}

For each KPI above, provide:
- The Power BI DAX formula
- The Tableau calculated field formula

Format clearly with the KPI name as a heading, then the two formulas
in code blocks. Assume standard column names from the list above."""

                    try:
                        formula_raw = cached_llm(formula_prompt)
                    except llm_client.QuotaExceeded:
                        formula_raw = None
                    if formula_raw is None:
                        st.warning(QUOTA_MESSAGE)
                    else:
                        formula_text = formula_raw.replace("```html", "").replace("```", "")
                        formula_text = html.unescape(formula_text)
                        # Remove common leading indentation and any leading newlines/spaces
                        formula_text = textwrap.dedent(formula_text).lstrip()
                        st.session_state.formula_result = formula_text

        if "formula_result" in st.session_state:
            st.subheader("KPI Formulas")
            st.markdown(st.session_state.formula_result, unsafe_allow_html=True)


    # ──────────────────────────────────────────────────────────────────
    # TAB 2 — Ask Questions
    # ──────────────────────────────────────────────────────────────────
    def _format_scalar(v):
        if isinstance(v, (bool, np.bool_)):
            return str(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return f"{v:,.2f}"
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d")
        return str(v)

    with tab_qa:
        if st.session_state.get("questions_failed"):
            st.warning(QUOTA_MESSAGE + " Suggested questions could not be generated yet.")
            if st.button("Try generating suggested questions again", key="retry_questions"):
                st.session_state.pop("last_uploaded_file", None)
                st.rerun()
        elif st.session_state.get("questions"):
            st.subheader("Click a question to get the answer")
            for i, q in enumerate(st.session_state.questions):
                if st.button(q, key=f"q_{i}"):
                    st.session_state.active_question = q
                    st.session_state.pop("last_answered_question", None)

        st.markdown("---")
        st.subheader("Or ask your own question")
        st.caption(
            "Questions must be about the loaded dataset. Anything unrelated or inappropriate "
            "returns \"Insufficient information/data\"."
        )

        typed_question = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g. Which region has the highest revenue?",
            max_chars=guardrails.MAX_QUESTION_LEN,
        )

        if st.button("Analyze", key="analyze_typed"):
            if typed_question.strip():
                st.session_state.active_question = typed_question
                st.session_state.pop("last_answered_question", None)
            else:
                st.warning("Please type a question first.")

        # ══════════════════════════════════════════════════════════════
        # ANSWER FLOW — only runs when a NEW question is activated
        # ══════════════════════════════════════════════════════════════
        current_q = st.session_state.get("active_question")
        if current_q and current_q != st.session_state.get("last_answered_question"):
            with st.spinner("Analyzing..."):
                try:
                    outcome = pipeline.run_question(df, current_q, cached_llm)
                except llm_client.QuotaExceeded:
                    outcome = {"status": "quota"}
                except Exception as e:  # noqa: BLE001 - never show a raw traceback to students
                    outcome = {
                        "status": "error",
                        "message": "Something went wrong while analyzing. Please try again or rephrase the question.",
                        "detail": f"{type(e).__name__}: {e}",
                    }
            st.session_state.last_answered_question = current_q
            st.session_state.last_answer = outcome
            st.session_state.answer_id = st.session_state.get("answer_id", 0) + 1

        # ── Display the persisted answer (survives re-runs) ──
        ans = st.session_state.get("last_answer")
        if ans:
            st.markdown("---")
            st.markdown(f"**Question:** {st.session_state.get('last_answered_question')}")
            status = ans.get("status")

            if status == "insufficient":
                st.warning(ans["message"])
            elif status == "quota":
                st.warning(QUOTA_MESSAGE)
            elif status == "error":
                st.error(ans["message"])
                detail = ans.get("detail") or ans.get("raw")
                if detail:
                    with st.expander("Technical details"):
                        st.code(str(detail))
            else:
                st.markdown("**Answer:**")
                st.write(ans["answer_text"])
                result = ans["result"]

                if ans["kind"] == "scalar":
                    st.metric("Result", _format_scalar(result))
                else:
                    # The AI/chart engine picks ONE chart that fits the question and the data.
                    fig = None
                    if ans.get("chart_type"):
                        fig = charts.build_figure(result, ans["chart_type"], ans.get("chart_title"))
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                    with st.expander("Show computed result", expanded=fig is None):
                        st.dataframe(result.head(1000), hide_index=True, width="stretch")

            if ans.get("code"):
                with st.expander("Show the code"):
                    st.code(ans["code"], language="python")
