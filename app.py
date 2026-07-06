import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
from industry_knowledge import INDUSTRY_KNOWLEDGE
import plotly.express as px
import json

# ── API key handling: works locally AND when deployed ──
load_dotenv()

api_key = None
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

# Page setup — must come before any other st.* calls
st.set_page_config(page_title="InsightPilot", layout="wide")

if not api_key:
    st.title("InsightPilot")
    st.info(
        "To use this app, enter your free Google Gemini API key below. "
        "Get one at https://aistudio.google.com/apikey"
    )
    api_key = st.text_input("Gemini API Key:", type="password")
    if not api_key:
        st.stop()

genai.configure(api_key=api_key)

st.title("InsightPilot")
st.write("AI-powered KPI advisor and analytics agent. Upload a CSV to begin.")

# ══════════════════════════════════════════════════════════════════
# SIDEBAR: Sample datasets + Clear button (always visible)
# ══════════════════════════════════════════════════════════════════
st.sidebar.header("📁 Try a sample")
st.sidebar.caption("Don't have a CSV? Load one of these to explore:")

sample_col1, sample_col2 = st.sidebar.columns(2)
with sample_col1:
    if st.button("🛒 Retail", use_container_width=True, key="sample_retail"):
        st.session_state.sample_file = "sample_data/retail_sales.csv"
        st.session_state.sample_name = "retail_sales.csv"
        # Clear question/answer state when switching to a new sample
        for k in ["active_question", "last_answered_question", "last_answer",
                  "kpi_result", "formula_result"]:
            st.session_state.pop(k, None)
with sample_col2:
    if st.button("🏦 Banking", use_container_width=True, key="sample_banking"):
        st.session_state.sample_file = "sample_data/banking_loans.csv"
        st.session_state.sample_name = "banking_loans.csv"
        for k in ["active_question", "last_answered_question", "last_answer",
                  "kpi_result", "formula_result"]:
            st.session_state.pop(k, None)

st.sidebar.markdown("---")

# Clear / Start Over button
if st.sidebar.button("🔄 Clear / Start Over", use_container_width=True):
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

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data_source_name = uploaded_file.name
    st.session_state.pop("sample_file", None)
elif "sample_file" in st.session_state:
    df = pd.read_csv(st.session_state.sample_file)
    data_source_name = st.session_state.sample_name

if df is not None:
    # ── Sidebar: file info + schema + preview ──
    st.sidebar.header("📄 Dataset")
    st.sidebar.write(f"**{data_source_name}**")
    st.sidebar.caption(f"{df.shape[0]} rows · {df.shape[1]} columns")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Schema")
    schema_df = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(t) for t in df.dtypes],
    })
    st.sidebar.dataframe(schema_df, hide_index=True, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Preview (first 5)")
    st.sidebar.dataframe(df.head(5), hide_index=True, use_container_width=True)

    # ── Auto-generate business questions when a new file/sample is loaded ──
    if st.session_state.get("last_uploaded_file") != data_source_name:
        with st.spinner("Reading your data and preparing insights..."):
            columns = list(df.columns)
            sample_rows = df.head(3).to_string()

            q_prompt = f"""You are a senior business analyst.

A dataset has these columns: {columns}
Sample rows:
{sample_rows}

Generate exactly 5 specific business questions a manager or stakeholder
would realistically ask about THIS data. Make them concrete and answerable
from the columns above (e.g. "Which region has the highest revenue?").

Return ONLY the 5 questions, one per line, no numbering, no extra text."""

            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            q_response = model.generate_content(q_prompt)

            questions = [q.strip() for q in q_response.text.split("\n") if q.strip()]
            st.session_state.questions = questions
            st.session_state.last_uploaded_file = data_source_name
            # Clear state from previous dataset
            for k in ["active_question", "last_answered_question", "last_answer",
                      "kpi_result", "formula_result"]:
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

                model = genai.GenerativeModel("gemini-2.5-flash-lite")
                response = model.generate_content(prompt)
                st.session_state.kpi_result = response.text

        if "kpi_result" in st.session_state:
            st.subheader("AI-Recommended KPIs")
            st.markdown(st.session_state.kpi_result)

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

                    model = genai.GenerativeModel("gemini-2.5-flash-lite")
                    formula_response = model.generate_content(formula_prompt)
                    st.session_state.formula_result = formula_response.text

        if "formula_result" in st.session_state:
            st.subheader("KPI Formulas")
            st.markdown(st.session_state.formula_result)

    # ──────────────────────────────────────────────────────────────────
    # TAB 2 — Ask Questions
    # ──────────────────────────────────────────────────────────────────
    with tab_qa:
        # ── Auto-generated question buttons ──
        if "questions" in st.session_state:
            st.subheader("Click a question to get the answer")
            for i, q in enumerate(st.session_state.questions):
                if st.button(q, key=f"q_{i}"):
                    st.session_state.active_question = q

        # ── Free-text question box ──
        st.markdown("---")
        st.subheader("Or ask your own question")

        typed_question = st.text_input(
            "Describe a business problem or ask anything about your data:",
            placeholder="e.g. Why might revenue be lower in some regions?",
        )

        if st.button("Analyze", key="analyze_typed"):
            if typed_question.strip():
                st.session_state.active_question = typed_question
            else:
                st.warning("Please type a question first.")

        # ══════════════════════════════════════════════════════════════
        # ANSWER FLOW — only runs when a NEW question is activated
        # (prevents burning API calls on every re-run)
        # ══════════════════════════════════════════════════════════════
        current_q = st.session_state.get("active_question")
        last_answered = st.session_state.get("last_answered_question")

        # Only fire the AI if the active question is NEW
        if current_q and current_q != last_answered:
            with st.spinner("Analyzing..."):
                column_info = ""
                for col in df.columns:
                    if df[col].dtype == "object":
                        unique_vals = df[col].unique()[:10]
                        column_info += f"- {col}: {list(unique_vals)}\n"
                    else:
                        column_info += f"- {col}: numeric ({df[col].dtype})\n"

                answer_prompt = f"""You are a Python pandas and Plotly expert.

A pandas DataFrame called `df` has these columns and their actual values:
{column_info}

Answer this question: "{current_q}"

Decide whether a chart would help. A single number or one-word answer does
NOT need a chart. A comparison, ranking, breakdown, or trend over time DOES.

Return ONLY a JSON object (no markdown, no backticks) with these keys:
- "answer": a short plain-English answer to the question
- "needs_chart": true or false
- "code": pandas code that computes the answer and stores it in a variable
  called `result`. If needs_chart is true, also build a Plotly figure stored
  in a variable called `fig` using plotly.express as px.

Rules for the code:
- `df`, `pd`, and `px` are already available. Do not import or load data.
- Use the EXACT column names and values shown above. Do not guess values.
- Return valid Python that runs as-is.
- If the question asks about a metric that cannot be computed from the
  available columns (e.g. profit margin without cost data), set
  "needs_chart" to false and set "answer" to a clear explanation that
  the data does not contain the columns needed, listing what would be
  required. Do NOT invent or approximate the metric. Set "code" to
  exactly: result = "Insufficient data"
"""

                model = genai.GenerativeModel("gemini-2.5-flash-lite")
                response = model.generate_content(answer_prompt)

                raw = response.text.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()

                try:
                    data = json.loads(raw)
                    answer_text = data.get("answer", "")
                    needs_chart = data.get("needs_chart", False)
                    code = data.get("code", "")

                    # ── Agentic retry loop ──
                    max_attempts = 3
                    local_vars = {"df": df, "pd": pd, "px": px}
                    real_result = None
                    last_error = None

                    for attempt in range(max_attempts):
                        try:
                            exec(code, {}, local_vars)
                            real_result = local_vars.get("result", None)
                            last_error = None
                            break

                        except Exception as e:
                            last_error = str(e)

                            if attempt == max_attempts - 1:
                                break

                            with st.spinner(f"Fixing the code (attempt {attempt + 2})..."):
                                fix_prompt = f"""The following pandas code failed with an error.

Code:
{code}

Error:
{last_error}

The DataFrame `df` has these columns and values:
{column_info}

Rewrite the code to fix the error. Use the EXACT column names and values
shown above. Store the answer in `result`. If a Plotly figure was being
built, keep it in a variable called `fig`.

Return ONLY the corrected Python code — no markdown, no backticks, no explanation."""

                                fix_response = model.generate_content(fix_prompt)
                                code = fix_response.text.strip()
                                code = code.replace("```python", "").replace("```", "").strip()
                                local_vars = {"df": df, "pd": pd, "px": px}

                    # Store the answer for persistent display across re-runs
                    st.session_state.last_answered_question = current_q
                    st.session_state.last_answer = {
                        "answer_text": answer_text,
                        "real_result": real_result,
                        "code": code,
                        "last_error": last_error,
                        "needs_chart": needs_chart,
                        "fig": local_vars.get("fig") if (needs_chart and "fig" in local_vars) else None,
                    }

                except Exception as e:
                    st.session_state.last_answered_question = current_q
                    st.session_state.last_answer = {
                        "answer_text": None,
                        "real_result": None,
                        "code": None,
                        "last_error": f"Could not parse AI response: {e}",
                        "needs_chart": False,
                        "fig": None,
                        "raw": raw,
                    }

        # ── Display the persisted answer (survives re-runs) ──
        if st.session_state.get("last_answer"):
            ans = st.session_state.last_answer
            st.markdown("---")
            st.markdown(f"**Question:** {st.session_state.last_answered_question}")

            if ans.get("last_error") and ans.get("answer_text") is None:
                st.error(ans["last_error"])
                if "raw" in ans:
                    with st.expander("Show raw AI response"):
                        st.code(ans["raw"])
            elif ans.get("last_error"):
                st.error(f"Could not run the analysis after 3 attempts. Last error: {ans['last_error']}")
            else:
                st.markdown("**Answer:**")
                st.write(ans["answer_text"])
                if ans.get("real_result") is not None:
                    st.markdown("**Computed result:**")
                    st.write(ans["real_result"])
                if ans.get("fig") is not None:
                    st.plotly_chart(ans["fig"], use_container_width=True)

            if ans.get("code"):
                with st.expander("Show the code"):
                    st.code(ans["code"], language="python")