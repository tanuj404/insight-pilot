"""
Guardrails for InsightPilot's "Ask Questions" feature.

Three layers (cheapest first):
  1. screen_question()  - local filter for empty/abusive/prompt-injection input.
                          Costs no API call.
  2. (in the LLM prompt) - the model must answer status="insufficient" for anything
                          that is not answerable from the loaded dataset.
  3. check_code() / safe_exec() - AST check + restricted namespace for the
                          model-written pandas code before it is executed.

NOTE: layer 3 is a best-effort classroom sandbox, not full OS-level isolation.
It removes the obvious ways to reach secrets, files or the network.
"""
import ast
import builtins
import re

import numpy as np
import pandas as pd
import plotly.express as px

MAX_QUESTION_LEN = 300
MIN_QUESTION_LEN = 3


def insufficient_message(columns=None):
    msg = (
        "**Insufficient information/data.** This question can't be answered from "
        "the loaded dataset, so no analysis was run."
    )
    columns = list(columns) if columns is not None else []
    if columns:
        shown = ", ".join(f"`{c}`" for c in columns[:8])
        msg += (
            f"\n\nTry a question about this data, for example about: {shown}"
            + (" ..." if len(columns) > 8 else "")
        )
    return msg


# ── Layer 1: local pre-filter ─────────────────────────────────────
_ABUSE_RE = re.compile(
    r"\b(fuck\w*|shit\w*|bitch\w*|asshole\w*|bastard\w*|cunt\w*|slut\w*|whore\w*|"
    r"idiot\w*|stupid|moron\w*|dumbass|retard\w*|piss\w*|crap|"
    r"kill yourself|kys|shut up|hate you|damn you)\b",
    re.IGNORECASE,
)

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+|any\s+|your\s+)?(previous|prior|above|earlier|preceding)",
    r"disregard\s+.{0,40}(instruction|rule|prompt)",
    r"(system|developer|hidden)\s+prompt",
    r"(reveal|show|print|display|tell|leak|give)\s+.{0,40}"
    r"(api[\s_-]?key|secret|password|token|credential|instruction|prompt)",
    r"api[\s_-]?key",
    r"os\.environ|st\.secrets|environ(ment)?\s+variable|\.env\b",
    r"__\w+__",
    r"you\s+are\s+now|pretend\s+(to\s+be|you)|jailbreak|developer\s+mode|\bdan\s+mode\b",
    r"\b(exec|eval|subprocess|__import__)\s*\(",
    r"\b(rm\s+-rf|drop\s+table|delete\s+from)\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def screen_question(question):
    """Return (ok, reason). reason is for logs/tests; the UI shows a generic message."""
    if question is None:
        return False, "empty"
    q = question.strip()
    if len(q) < MIN_QUESTION_LEN:
        return False, "too_short"
    if len(q) > MAX_QUESTION_LEN:
        return False, "too_long"
    if len(re.findall(r"[A-Za-z]", q)) < 2:
        return False, "no_text"
    if _ABUSE_RE.search(q):
        return False, "abusive"
    if _INJECTION_RE.search(q):
        return False, "prompt_injection"
    return True, "ok"


# ── Layer 3: code safety ──────────────────────────────────────────
_FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "shutil", "pathlib", "socket", "requests", "urllib",
    "importlib", "builtins", "open", "eval", "exec", "compile", "__import__",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr", "hasattr",
    "input", "breakpoint", "exit", "quit", "help", "memoryview", "type", "super",
    "st", "streamlit", "genai", "environ", "secrets", "pickle", "ctypes",
}

_FORBIDDEN_ATTRS = _FORBIDDEN_NAMES | {
    # pandas I/O that can read/write files or hit the network
    "read_csv", "read_excel", "read_json", "read_pickle", "read_sql", "read_html",
    "read_parquet", "read_table", "read_clipboard", "read_fwf", "read_xml",
    "read_feather", "read_hdf", "read_orc", "read_sas", "read_spss", "read_stata",
    "to_csv", "to_excel", "to_pickle", "to_sql", "to_parquet", "to_json",
    "to_clipboard", "to_hdf", "to_feather", "to_xml", "to_orc", "to_stata",
    "to_latex", "to_markdown",
    "system", "popen", "mro", "load", "save", "savetxt", "loadtxt", "fromfile",
    "tofile", "write", "eval",
}

_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "int", "isinstance", "len", "list", "map", "max", "min", "pow", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "True", "False", "None", "ValueError", "KeyError", "TypeError", "Exception",
]


def _safe_range(*args):
    r = range(*args)
    if len(r) > 1_000_000:
        raise ValueError("range too large")
    return r


def _safe_builtins():
    safe = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)}
    safe["range"] = _safe_range
    return safe


def check_code(code):
    """Return (ok, reason) after statically checking model-written code."""
    if not code or not code.strip():
        return False, "empty code"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "imports are not allowed"
        if isinstance(node, (ast.While, ast.AsyncFunctionDef, ast.Await, ast.Global,
                             ast.Nonlocal, ast.ClassDef, ast.Try)):
            return False, f"{type(node).__name__} is not allowed"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"use of '{node.id}' is not allowed"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return False, f"access to '{node.attr}' is not allowed"
            if node.attr in _FORBIDDEN_ATTRS:
                return False, f"use of '.{node.attr}' is not allowed"
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            return False, "double-underscore strings are not allowed"
    return True, "ok"


def safe_exec(code, df):
    """
    Run checked code in a restricted namespace and return that namespace.
    A single dict is used for globals so comprehensions/lambdas can see variables.
    The DataFrame is copied so the user's loaded data is never modified.
    """
    ok, reason = check_code(code)
    if not ok:
        raise PermissionError(reason)
    namespace = {
        "__builtins__": _safe_builtins(),
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "px": px,
    }
    exec(compile(code, "<analysis>", "exec"), namespace)
    return namespace
