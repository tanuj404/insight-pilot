"""
Robust CSV loading + light automatic cleaning, so real-world (e.g. Kaggle) files
work without students having to fix them first.

  read_csv_robust()  - handles encodings (Kaggle files are often latin-1) and ; / tab separators
  prepare_dataframe() - tidy column names, parse date columns, convert numbers stored as text
"""
import re
import warnings

import pandas as pd

_DATE_NAME_RE = re.compile(r"(date|time|month|year|day|week|quarter|period|timestamp)", re.IGNORECASE)
_DATE_VALUE_RE = re.compile(r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
                            r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})")


def read_csv_robust(source):
    """Read a CSV from a path or an uploaded file object, trying common encodings."""
    last_err = None
    df = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
    if df is None:
        raise last_err

    # A single wide column usually means the file is ; or tab separated.
    if df.shape[1] == 1:
        header = str(df.columns[0])
        for sep in (";", "\t", "|"):
            if sep in header:
                if hasattr(source, "seek"):
                    source.seek(0)
                df = pd.read_csv(source, encoding=enc, sep=sep)
                break
    return df


def _is_text(series):
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _try_datetime(series):
    """Return a datetime Series if >=90% of non-null values parse, else None."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() >= 0.9 * len(non_null):
        return parsed
    return None


def prepare_dataframe(df):
    """Return (clean_df, notes). `notes` lists what was automatically adjusted."""
    notes = []
    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    # Drop an exported-index column ("Unnamed: 0")
    for col in [c for c in df.columns if c.lower().startswith("unnamed")]:
        if df[col].isna().all() or (pd.api.types.is_integer_dtype(df[col]) and df[col].is_unique):
            df = df.drop(columns=col)
            notes.append(f"Dropped empty/index column '{col}'")

    # Drop fully empty columns
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        df = df.drop(columns=empty)
        notes.append(f"Dropped empty columns: {', '.join(empty)}")

    for col in list(df.columns):
        s = df[col]
        if not _is_text(s):
            continue

        s = s.astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.replace("", pd.NA)
        non_null = s.dropna()
        if non_null.empty:
            df[col] = s
            continue

        # Numbers stored as text (e.g. Telco "TotalCharges" with blank cells)
        as_num = pd.to_numeric(non_null.astype(str).str.replace(",", "", regex=False), errors="coerce")
        looks_like_date = bool(_DATE_NAME_RE.search(col)) and bool(
            _DATE_VALUE_RE.match(str(non_null.iloc[0]))
        )
        if not looks_like_date and as_num.notna().mean() >= 0.9:
            cleaned = s.map(lambda v: v.replace(",", "") if isinstance(v, str) else v)
            converted = pd.to_numeric(cleaned, errors="coerce")
            bad = int(converted.isna().sum() - s.isna().sum())
            df[col] = converted
            notes.append(
                f"Converted '{col}' to numbers" + (f" ({bad} unreadable value(s) left blank)" if bad else "")
            )
            continue

        # Date columns stored as text
        sample = non_null.astype(str).head(20)
        if _DATE_NAME_RE.search(col) or sample.map(lambda v: bool(_DATE_VALUE_RE.match(v))).mean() >= 0.8:
            parsed = _try_datetime(s)
            if parsed is not None:
                df[col] = parsed
                notes.append(f"Parsed '{col}' as a date")
                continue

        df[col] = s

    return df, notes


def dtype_label(series):
    """Short, student-friendly dtype label for the Schema panel."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_numeric_dtype(series):
        return "decimal"
    return "text"
