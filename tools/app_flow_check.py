"""
Drive the REAL app.py headlessly (Streamlit AppTest) with a dataset and a list of questions,
and report status + chart type for each. Uses the real AI (GEMINI_API_KEY from .env).

    venv\\Scripts\\python.exe tools\\app_flow_check.py sample_data/retail_sales.csv "question 1" "question 2" ...
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest


def main():
    path = sys.argv[1]
    questions = sys.argv[2:]

    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    print("landing page OK (no exceptions)")

    at.session_state["sample_file"] = path
    at.session_state["sample_name"] = os.path.basename(path)
    at.run()
    assert not at.exception, at.exception
    print(f"dataset loaded: {os.path.basename(path)}; starter questions: {len(at.session_state['questions'])}")
    for q in at.session_state["questions"]:
        print("   -", q)

    for q in questions:
        at.text_input[0].set_value(q)
        at.run()
        next(b for b in at.button if b.key == "analyze_typed").click()
        at.run()
        assert not at.exception, at.exception
        ans = at.session_state["last_answer"]
        status = ans.get("status")
        detail = ""
        if status == "ok":
            if ans["kind"] == "table":
                detail = f"table{ans['result'].shape} chart={ans['chart_type']} cols={list(ans['result'].columns)}"
            else:
                detail = f"scalar={ans['result']}"
            detail += f"\n        answer: {ans['answer_text'][:160]}"
        elif status == "error":
            detail = ans.get("detail") or ans.get("message")
        shown_warning = [a.value[:60] for a in at.warning]
        print(f"\n[{status}] {q}\n        {detail}")
        if status == "insufficient":
            print(f"        warning shown: {shown_warning[:1]}")

    # also exercise the starter-question buttons (first one)
    qb = next(b for b in at.button if b.key == "q_0")
    qb.click()
    at.run()
    assert not at.exception, at.exception
    ans = at.session_state["last_answer"]
    print(f"\nstarter button 1 -> status={ans['status']} chart={ans.get('chart_type')}")


if __name__ == "__main__":
    main()
