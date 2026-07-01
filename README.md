# InsightPilot

**An AI agent that recommends KPIs, generates DAX & Tableau formulas, and answers analyst questions for any CSV — across 21 industries.**

Built with Python, Streamlit, and Google Gemini.

---

## The problem

When a business or data analyst gets a new dataset, the first hours are spent on repetitive setup: figuring out which KPIs matter for the industry, writing the DAX or Tableau formulas from scratch, and guessing at the questions stakeholders will ask. Most tools built on top of LLMs solve *half* of this — you can already chat with a CSV in ChatGPT — but none of them proactively tell an analyst *what to track*, *how to compute it in Power BI or Tableau*, and *what a manager is likely to ask next*.

InsightPilot does that.

---

## What it does

Upload a CSV, pick an industry, and InsightPilot produces:

- **KPI recommendations** grounded in a curated industry knowledge base *and* the AI's own analysis of the actual columns
- **Power BI DAX formulas** and **Tableau calculated field formulas** for each KPI
- **Auto-generated business questions** stakeholders would realistically ask
- **Executable answers with charts** — the AI writes pandas + Plotly code, the app runs it locally on your data, and shows the result

Two modes, one clean interface:

| Mode | Purpose |
|---|---|
| 🎯 **KPI Briefing** | Recommend KPIs and formulas for your industry |
| 💬 **Ask Questions** | Click a suggested question or type your own; get an answer plus a chart |

---

## Architecture

The model choice matters less than the engineering around it. The core design:

- **Knowledge base** — a curated dictionary of industry-specific KPIs and business problems, used as prompt context (`industry_knowledge.py`)
- **Grounding layer** — every prompt injects the actual column names, types, and unique values so the model writes code against reality, not assumptions
- **Structured output (JSON)** — the model returns labeled fields (`answer`, `needs_chart`, `code`) instead of free text
- **Local code execution** — the model writes pandas / Plotly code; the app runs it on the user's data. The data never leaves the machine.
- **Agentic retry loop** — when generated code crashes, the error is fed back to the model for self-correction, capped at 3 attempts
- **Refusal guardrail** — the prompt instructs the model to reply *"Insufficient data"* rather than fabricate metrics the columns can't support

The model is interchangeable. The architecture is the interesting part.

---

## Tech stack

- **Python 3.11**
- **Streamlit** — web UI
- **pandas** — data manipulation
- **Google Gemini** (`google-generativeai`) — LLM
- **Plotly Express** — interactive charts
- **python-dotenv** — secrets management

---

## Run it locally

**Prerequisites:** Python 3.11+, a free [Google AI Studio API key](https://aistudio.google.com/apikey).

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/insight-pilot.git
cd insight-pilot

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
# Create a file called .env in the project root:
echo GEMINI_API_KEY=your_key_here > .env

# 5. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Project structure

```
insight-pilot/
├── app.py                    # Main Streamlit app
├── industry_knowledge.py     # Curated KPI knowledge base
├── requirements.txt          # Python dependencies
├── .env                      # API key (not committed)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Screenshots

*Add your screenshots here once deployed. Suggested shots:*

- Landing page with sidebar showing schema and preview
- KPI Briefing tab with generated KPIs
- DAX and Tableau formulas
- Ask Questions tab with an answered question and chart

---

## Roadmap

- [ ] Auto Insights tab — surface top findings without user prompting
- [ ] Data quality checks — flag missing values, duplicates, date parsing issues
- [ ] Multi-turn chat memory
- [ ] Export briefing as PDF
- [ ] "Bring your own API key" input for public deployment

---

## Known limitations

- LLM-generated DAX is not exhaustively validated — always test formulas in Power BI before shipping to stakeholders.
- Runs on Gemini's free tier; quota-limited without a paid key.
- Code is executed with Python's `exec()` — safe for local use with your own CSVs, not intended as a hosted multi-tenant service without sandboxing.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

Built by **Tanuj Ahuja** — MS Business Analytics student, prior Business Analyst experience in Power BI, DAX, SQL, and Python.

GitHub: [@YOUR-USERNAME](https://github.com/YOUR-USERNAME)
