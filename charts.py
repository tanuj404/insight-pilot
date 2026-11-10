"""
Chart selection + rendering for InsightPilot.

The model only returns pandas code that produces a `result`. This module then looks at
the SHAPE of that result (dates? categories? numbers?) and the wording of the question
and decides which chart fits:

    time / date involved ............ line (or area / column)
    ranking, many categories ........ horizontal bar, sorted
    share of a whole, <= 6 groups ... pie / donut
    two category dimensions ......... heatmap
    two numeric measures ............ scatter (+ trend line)
    one numeric column .............. histogram / box plot
    a single number ................. KPI card (no chart)

The model's own suggestion (`hint`) is only used when it is valid for the data.
"""
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#1E3A8A", "#2563EB", "#0EA5E9", "#14B8A6", "#F59E0B", "#EF4444", "#8B5CF6", "#64748B"]

CHART_LABELS = {
    "line": "Line chart",
    "area": "Area chart",
    "bar": "Column chart",
    "barh": "Horizontal bar chart",
    "pie": "Pie / donut chart",
    "heatmap": "Heatmap",
    "scatter": "Scatter plot",
    "histogram": "Histogram",
    "box": "Box plot",
}

VALID_HINTS = set(CHART_LABELS) | {"none"}

_TIME_RE = re.compile(
    r"\b(over time|trend\w*|monthly|daily|weekly|yearly|quarterly|annual\w*|per (month|year|week|day|quarter)|"
    r"by (month|year|week|day|quarter|date)|each (month|year|week|day|quarter)|time series|"
    r"growth|season\w*|timeline|year[- ]over[- ]year|month[- ]over[- ]month|date|dates)\b",
    re.IGNORECASE,
)
_SHARE_RE = re.compile(
    r"(\b(share|proportion|percent\w*|composition|breakdown|split|portion|make[s]? up|contribut\w*)\b|%)",
    re.IGNORECASE,
)
_RANK_RE = re.compile(
    r"\b(top|bottom|highest|lowest|best|worst|rank\w*|most|least|largest|smallest|leading)\b",
    re.IGNORECASE,
)
_RELATION_RE = re.compile(
    r"\b(correlat\w*|relationship|versus|vs\.?|against|impact of|affect\w*|influenc\w*)\b",
    re.IGNORECASE,
)

_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_MONTHS_ABBR = [m[:3] for m in _MONTHS]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAYS_ABBR = [d[:3] for d in _WEEKDAYS]

_TIMELIKE_NAME_RE = re.compile(
    r"^(order[_ ]?|ship[_ ]?|sale[_ ]?|fiscal[_ ]?|calendar[_ ]?)?(year|month|quarter|week|day|hour)([_ ]?(num|number|no))?$",
    re.IGNORECASE,
)


# ── helpers ───────────────────────────────────────────────────────
def _calendar_rank(values):
    """If all values are month/weekday names return {value: rank}, else None."""
    labels = [str(v).strip().lower() for v in values]
    for full, abbr in ((_MONTHS, _MONTHS_ABBR), (_WEEKDAYS, _WEEKDAYS_ABBR)):
        if all(l in full or l in abbr or l[:3] in abbr for l in labels) and len(set(labels)) > 1:
            return {v: (full.index(l) if l in full else abbr.index(l[:3]))
                    for v, l in zip(values, labels)}
    return None


def _is_timelike_numeric(name, series):
    if not _TIMELIKE_NAME_RE.match(str(name).strip()):
        return False
    s = series.dropna()
    return len(s) > 0 and bool((s % 1 == 0).all())


def split_columns(rdf):
    """Split columns into (time_dims, category_dims, numeric_measures)."""
    time, cat, nums = [], [], []
    for c in rdf.columns:
        s = rdf[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            time.append(c)
        elif pd.api.types.is_bool_dtype(s):
            cat.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            (time if _is_timelike_numeric(c, s) else nums).append(c)
        else:
            cat.append(c)
    return time, cat, nums


# ── result normalisation ──────────────────────────────────────────
def _is_scalar(x):
    return isinstance(x, (int, float, str, bool, np.integer, np.floating, np.bool_, pd.Timestamp))


def prepare_result(result):
    """
    Normalise whatever the analysis returned.
    Returns ("none", None) | ("scalar", value) | ("table", DataFrame)
    """
    if result is None:
        return "none", None
    if _is_scalar(result):
        return "scalar", result
    if isinstance(result, dict):
        result = pd.Series(result)
    elif isinstance(result, (list, tuple, np.ndarray)):
        result = pd.Series(list(result))

    if isinstance(result, pd.Series):
        s = result.copy()
        name = str(s.name) if s.name is not None else "value"
        index_names = list(s.index.names)
        default_index = isinstance(s.index, pd.RangeIndex) or (
            index_names == [None] and pd.api.types.is_integer_dtype(s.index)
        )
        if default_index:
            frame = s.rename(name).to_frame()
        else:
            if name in index_names:
                name = f"{name}_value"
            frame = s.rename(name).reset_index()
            if index_names == [None] and "index" in frame.columns:
                frame = frame.rename(columns={"index": "category"})
    elif isinstance(result, pd.DataFrame):
        frame = result.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [" ".join(str(p) for p in c).strip() for c in frame.columns]
        frame.columns.name = None
        names = list(frame.index.names)
        unnamed_int = names == [None] and pd.api.types.is_integer_dtype(frame.index)
        if not (isinstance(frame.index, pd.RangeIndex) or unnamed_int):
            try:
                frame = frame.reset_index()
                if names == [None] and "index" in frame.columns:
                    frame = frame.rename(columns={"index": "category"})
            except ValueError:
                frame = frame.reset_index(drop=True)
        else:
            frame = frame.reset_index(drop=True)
    else:
        return "scalar", str(result)

    frame.columns = [str(c) for c in frame.columns]
    for c in frame.columns:
        if isinstance(frame[c].dtype, pd.PeriodDtype):
            frame[c] = frame[c].dt.to_timestamp()
    if frame.empty:
        return "none", None
    if frame.shape == (1, 1):
        return "scalar", frame.iat[0, 0]
    return "table", frame


# ── choosing the chart ────────────────────────────────────────────
def chart_options(rdf, question="", hint=None):
    """All chart types that make sense for this result (unordered preference handled elsewhere)."""
    time, cat, nums = split_columns(rdf)
    dims = time + cat
    n = len(rdf)
    opts = []
    if not nums or n < 2:
        return opts

    if len(dims) == 0:
        if len(nums) >= 2 and n >= 3:
            opts.append("scatter")
        if n >= 5:
            opts += ["histogram", "box"]
    elif len(dims) == 1:
        d = dims[0]
        k = rdf[d].nunique()
        if d in time:
            if n >= 2:
                opts += ["line", "area", "bar"]
        else:
            if _calendar_rank(rdf[d].tolist()) and n >= 2:
                opts.append("line")
            opts += ["bar", "barh"]
            if 2 <= k <= 12:
                opts.append("pie")
        if hint == "heatmap" and len(nums) >= 2:
            opts.append("heatmap")
    else:
        if time:
            opts.append("line")
        if rdf.drop_duplicates(subset=dims[:2]).shape[0] <= 2500:
            opts.append("heatmap")
        opts.append("bar")

    seen = set()
    return [o for o in opts if not (o in seen or seen.add(o))]


def choose_chart_type(rdf, question="", hint=None):
    """Pick the best chart for this result. Returns a chart-type string or None."""
    opts = chart_options(rdf, question, hint)
    if not opts:
        return None
    time, cat, nums = split_columns(rdf)
    dims = time + cat
    q = question or ""

    # Rule 1: anything time-based -> line chart.
    if "line" in opts and (time or _TIME_RE.search(q)):
        return "line"

    # Rule 2: two category dimensions -> heatmap when the grid is big enough to read as one,
    # otherwise grouped columns (a 4 x 2 grid is clearer as bars).
    if len(dims) >= 2:
        sizes = [rdf[d].nunique() for d in dims[:2]]
        if "heatmap" in opts and min(sizes) >= 3:
            return "heatmap"
        return "bar"

    # Rule 3: the model's suggestion, if valid for this data.
    if hint in opts:
        return hint

    # Rule 4: fall back to shape-based rules.
    if len(dims) == 0:
        if "scatter" in opts and (_RELATION_RE.search(q) or "histogram" not in opts):
            return "scatter"
        return "histogram" if "histogram" in opts else opts[0]
    if len(dims) == 1:
        k = rdf[dims[0]].nunique()
        if "pie" in opts and k <= 6 and _SHARE_RE.search(q):
            return "pie"
        if _RANK_RE.search(q) or k > 6:
            return "barh"
        return "bar"
    return opts[0]


# ── rendering ─────────────────────────────────────────────────────
def _style(fig, title=None, height=440):
    fig.update_layout(
        template="plotly_white",
        colorway=PALETTE,
        title=dict(text=title, x=0, font=dict(size=16)) if title else None,
        height=height,
        margin=dict(l=10, r=10, t=60 if title else 20, b=10),
        font=dict(size=13),
    )
    if not (fig.data and fig.data[0].type == "pie"):
        fig.update_layout(showlegend=len(fig.data) > 1)
    return fig


def _label_dates(series):
    """Datetime -> ISO-style string, monthly if every date is the 1st."""
    if (series.dt.day == 1).all():
        return series.dt.strftime("%Y-%m")
    return series.dt.strftime("%Y-%m-%d")


def _ordered_labels(values):
    rank = _calendar_rank(list(values))
    if rank:
        return sorted(values, key=lambda v: rank[v])
    return list(values)


def build_figure(rdf, chart_type, title=None):
    """Build a Plotly figure for `rdf` as `chart_type`. Returns a figure or None."""
    time, cat, nums = split_columns(rdf)
    dims = time + cat
    rdf = rdf.copy()
    fig = None

    if chart_type in ("line", "area") and dims and nums:
        x = dims[0]
        color = dims[1] if len(dims) > 1 else None
        if x in cat:
            rank = _calendar_rank(rdf[x].tolist())
            rdf = rdf.assign(_k=rdf[x].map(rank)).sort_values("_k").drop(columns="_k") if rank else rdf
        else:
            rdf = rdf.sort_values(x)
        if x in time and not pd.api.types.is_datetime64_any_dtype(rdf[x]):
            rdf[x] = rdf[x].astype(int).astype(str)
        y = nums[0] if (color or len(nums) == 1) else nums
        if chart_type == "line":
            fig = px.line(rdf, x=x, y=y, color=color, markers=len(rdf) <= 60)
        else:
            fig = px.area(rdf, x=x, y=y, color=color)

    elif chart_type == "bar" and dims and nums:
        x = dims[0]
        color = dims[1] if len(dims) > 1 else None
        if x in cat:
            rank = _calendar_rank(rdf[x].tolist())
            if rank:
                rdf = rdf.assign(_k=rdf[x].map(rank)).sort_values("_k").drop(columns="_k")
            elif not color:
                rdf = rdf.sort_values(nums[0], ascending=False)
        else:
            rdf = rdf.sort_values(x)
            if not pd.api.types.is_datetime64_any_dtype(rdf[x]):
                rdf[x] = rdf[x].astype(int).astype(str)
        y = nums[0] if (color or len(nums) == 1) else nums
        fig = px.bar(rdf, x=x, y=y, color=color, barmode="group", text_auto=".3s")

    elif chart_type == "barh" and dims and nums:
        d = dims[0]
        color = dims[1] if len(dims) > 1 else None
        rdf = rdf.sort_values(nums[0], ascending=True)
        suffix = ""
        if len(rdf) > 25:
            rdf = rdf.tail(25)
            suffix = " (top 25)"
        if d in time and not pd.api.types.is_datetime64_any_dtype(rdf[d]):
            rdf[d] = rdf[d].astype(int).astype(str)
        x = nums[0] if (color or len(nums) == 1) else nums
        fig = px.bar(rdf, x=x, y=d, color=color, orientation="h", barmode="group", text_auto=".3s")
        fig.update_yaxes(categoryorder="total ascending" if not color else "trace")
        if title:
            title += suffix

    elif chart_type == "pie" and dims and nums:
        d, v = dims[0], nums[0]
        data = rdf[[d, v]].groupby(d, as_index=False)[v].sum().sort_values(v, ascending=False)
        if len(data) > 8:
            other = data.iloc[7:][v].sum()
            data = pd.concat([data.iloc[:7], pd.DataFrame({d: ["Other"], v: [other]})], ignore_index=True)
        fig = px.pie(data, names=d, values=v, hole=0.45)
        fig.update_traces(textinfo="percent+label", sort=False)
        fig.update_layout(showlegend=len(data) > 6)

    elif chart_type == "heatmap":
        if len(dims) >= 2 and nums:
            r, c = dims[0], dims[1]
            for col in (r, c):
                if pd.api.types.is_datetime64_any_dtype(rdf[col]):
                    rdf[col] = _label_dates(rdf[col])
                elif col in time:
                    rdf[col] = rdf[col].astype(int).astype(str)
            pivot = rdf.pivot_table(index=r, columns=c, values=nums[0], aggfunc="sum")
            pivot = pivot.reindex(index=_ordered_labels(pivot.index), columns=_ordered_labels(pivot.columns))
            if pivot.shape[0] > 40:
                pivot = pivot.loc[pivot.sum(axis=1).nlargest(40).index]
            if pivot.shape[1] > 40:
                pivot = pivot[pivot.sum(axis=0).nlargest(40).index]
            fig = px.imshow(pivot, text_auto=".3s", aspect="auto", color_continuous_scale="Blues")
            fig.update_xaxes(title=str(c), side="bottom")
            fig.update_yaxes(title=str(r))
            fig.update_coloraxes(colorbar_title=nums[0])
        elif len(dims) == 1 and len(nums) >= 2:
            pivot = rdf.set_index(dims[0])[nums]
            fig = px.imshow(pivot, text_auto=".3s", aspect="auto", color_continuous_scale="Blues")

    elif chart_type == "scatter" and len(nums) >= 2:
        color = cat[0] if cat and rdf[cat[0]].nunique() <= 10 else None
        # Plot at most 5,000 points so large datasets stay fast; the trend line uses all rows.
        plot_df = rdf.sample(5000, random_state=0) if len(rdf) > 5000 else rdf
        fig = px.scatter(plot_df, x=nums[0], y=nums[1], color=color, opacity=0.6)
        valid = rdf[[nums[0], nums[1]]].dropna()
        if len(valid) >= 3 and valid[nums[0]].nunique() > 1:
            slope, intercept = np.polyfit(valid[nums[0]], valid[nums[1]], 1)
            xs = np.array([valid[nums[0]].min(), valid[nums[0]].max()])
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                     name="Trend line", line=dict(color="#EF4444", dash="dash")))

    elif chart_type == "histogram" and nums:
        color = cat[0] if cat and rdf[cat[0]].nunique() <= 6 else None
        fig = px.histogram(rdf, x=nums[0], color=color, nbins=30, barmode="overlay", opacity=0.85)

    elif chart_type == "box" and nums:
        x = cat[0] if cat and rdf[cat[0]].nunique() <= 12 else None
        fig = px.box(rdf, y=nums[0], x=x, points="outliers")

    if fig is None:
        return None
    if title is None and nums:
        title = f"{nums[0]} by {dims[0]}" if dims else nums[0]
    return _style(fig, title)
