
# app_with_call_talktime_report.py
# Streamlit app scaffold + "Call Talk-time Report" (Performance) — drop-in full file
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, time
from calendar import monthrange

# ======================
# Page & minimal styling
# ======================
st.set_page_config(page_title="JetLearn – Performance & Calls", page_icon="📞", layout="wide")
st.markdown(
    """
    <style>
      .stMetric { border: 1px solid #e5e7eb; border-radius: 12px; padding: 8px; background: #fff; }
      .stRadio > div { gap: 10px; }
      .legend-pill { display: inline-block; padding: 6px 12px; border-radius: 999px; margin-right: 10px;
                     font-weight: 600; font-size: 0.9rem; color: #111827; background: #e5e7eb; }
      .note { color:#374151; font-size:.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================
# Helpers
# ======================
def month_bounds(any_date: date):
    """Return (first_day, last_day) for the month of any_date."""
    y, m = any_date.year, any_date.month
    return date(y, m, 1), date(y, m, monthrange(y, m)[1])

def _find_col(df: pd.DataFrame, candidates: list[str]):
    if df is None or df.empty:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

def _parse_duration_to_seconds(x) -> float:
    """
    Accepts seconds as number/string OR HH:MM:SS / MM:SS.
    Returns seconds (float). Unparseable -> 0.
    """
    if pd.isna(x):
        return 0.0
    try:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        s = str(x).strip()
        if s.isdigit():
            return float(s)
        parts = [p.strip() for p in s.split(":") if p.strip() != ""]
        if len(parts) == 3:   # HH:MM:SS
            h, m, sec = parts
            return float(int(h) * 3600 + int(m) * 60 + int(float(sec)))
        if len(parts) == 2:   # MM:SS
            m, sec = parts
            return float(int(m) * 60 + int(float(sec)))
    except Exception:
        pass
    return 0.0

def _seconds_to_hms(total_seconds: float) -> str:
    total_seconds = int(round(float(total_seconds or 0)))
    h = total_seconds // 3600
    rem = total_seconds % 3600
    m = rem // 60
    s = rem % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

@st.cache_data(show_spinner=False)
def load_calls_csv(path: str) -> pd.DataFrame:
    dfc = pd.read_csv(path, low_memory=False)
    dfc.columns = [c.strip() for c in dfc.columns]
    # Map expected columns robustly
    dt_col = _find_col(dfc, [
        "Call Start Time","Start Time","Call Start","Call Time","Start Datetime","Date Time","Datetime","Date/Time"
    ])
    caller_col = _find_col(dfc, ["Caller","Agent","User","Owner","Owner Name","Created By"])
    ctype_col  = _find_col(dfc, ["Call Type","Type","Direction"])
    country_col = _find_col(dfc, ["Country Name","Country"])
    dur_col    = _find_col(dfc, ["Call Duration","Duration","Talk Time","Talk-Time","Duration (s)"])
    # Coerce DT + duration
    if dt_col:
        # Try both DMY/MDY — infer and be lenient
        dfc["_dt"] = pd.to_datetime(dfc[dt_col], errors="coerce", infer_datetime_format=True, dayfirst=True)
        # If many NaT, retry without dayfirst
        if dfc["_dt"].isna().mean() > 0.5:
            dfc["_dt"] = pd.to_datetime(dfc[dt_col], errors="coerce", infer_datetime_format=True, dayfirst=False)
    else:
        dfc["_dt"] = pd.NaT
    if dur_col:
        dfc["_secs"] = dfc[dur_col].apply(_parse_duration_to_seconds)
    else:
        dfc["_secs"] = 0.0
    # Friendly accessors
    dfc["_caller"]  = (dfc[caller_col].astype(str) if caller_col else pd.Series("Unknown", index=dfc.index))
    dfc["_ctype"]   = (dfc[ctype_col].astype(str)  if ctype_col  else pd.Series("Unknown", index=dfc.index))
    dfc["_country"] = (dfc[country_col].astype(str) if country_col else pd.Series("Unknown", index=dfc.index))
    dfc["_date"] = dfc["_dt"].dt.date
    dfc["_time"] = dfc["_dt"].dt.time
    return dfc

def _opts(series: pd.Series):
    if series is None or series.empty:
        return ["All"]
    vals = sorted({str(v) for v in series.dropna().unique()})
    return ["All"] + list(vals)

# ======================
# Sidebar — Data source
# ======================
with st.sidebar.expander("Call Talk-time • Data Source", expanded=False):
    st.caption("Select the CSV that contains your call activity export.")
    calls_default = "/mnt/data/activityFeedReport_downloads641023449ff5870ffa44af631759841842.csv"
    calls_path = st.text_input("Calls CSV path", value=calls_default, key="calls_csv_path")

# ======================
# Navigation
# ======================
MASTER_SECTIONS = {
    "Performance": [
        "Call Talk-time Report",
        # Below are placeholders; replace with your real views if you want to keep all in one file:
        "MIS","Daily Business","Sales Tracker","AC Wise Detail","Leaderboard"
    ],
    "Funnel & Movement": ["Funnel","Lead Movement","Stuck deals","Deal Velocity","Deal Decay","Carry Forward"],
    "Insights & Forecast": ["Predictibility","Business Projection","Buying Propensity","80-20","Trend & Analysis","Heatmap","Bubble Explorer","Master Graph"],
    "Marketing": ["Referrals","HubSpot Deal Score tracker","Marketing Lead Performance & Requirement"],
}

# Default selection
sec_keys = list(MASTER_SECTIONS.keys())
default_section = sec_keys[0]
default_view = MASTER_SECTIONS[default_section][0]

# Sidebar Nav
st.sidebar.markdown("### Navigation")
section = st.sidebar.selectbox("Section", options=sec_keys, index=0, key="nav_section")
view = st.sidebar.selectbox("View", options=MASTER_SECTIONS[section], index=0, key="nav_view")

# Title bar
st.title("JetLearn — Performance & Calls")

# ======================
# View: Call Talk-time Report
# ======================
if view == "Call Talk-time Report":
    st.subheader("Performance — Call Talk-time Report")

    # Load calls
    try:
        calls_df = load_calls_csv(st.session_state.get("calls_csv_path", calls_path))
        if calls_df.empty:
            st.info("Calls CSV loaded but appears empty. Check the file.")
    except Exception as e:
        st.error(f"Could not load calls CSV. {e}")
        calls_df = pd.DataFrame()

    if not calls_df.empty:
        # ----- Filters -----
        filt_mode = st.radio("Date selection", ["Month", "Custom range"], index=0, horizontal=True)

        today = date.today()
        if filt_mode == "Month":
            sel_month = st.date_input("Pick any date in the target month", value=today.replace(day=1))
            m_start, m_end = month_bounds(sel_month)
            date_start, date_end = m_start, m_end
        else:
            c1, c2 = st.columns(2)
            with c1:
                date_start = st.date_input("Start date", value=today.replace(day=1))
            with c2:
                date_end   = st.date_input("End date (inclusive)", value=today)
            if date_end < date_start:
                st.error("End date cannot be before start date.")
                st.stop()

        t1, t2 = st.columns(2)
        with t1:
            start_time = st.time_input("Start time (daily)", value=time(0,0,0))
        with t2:
            end_time   = st.time_input("End time (daily)", value=time(23,59,59))

        col1, col2, col3 = st.columns(3)
        with col1:
            sel_callers = st.multiselect("Caller", options=_opts(calls_df["_caller"]), default=["All"])
        with col2:
            sel_ctypes = st.multiselect("Call Type", options=_opts(calls_df["_ctype"]), default=["All"])
        with col3:
            sel_countries = st.multiselect("Country Name", options=_opts(calls_df["_country"]), default=["All"])

        # ----- Apply filters -----
        dfc = calls_df.copy()
        # date filter
        mask_date = dfc["_date"].between(date_start, date_end)
        # time-of-day (handle NaT safely)
        time_series = dfc["_dt"].dt.time
        mask_time = time_series.ge(start_time) & time_series.le(end_time)
        mask = mask_date & mask_time

        # Caller
        if sel_callers and "All" not in sel_callers:
            mask = mask & dfc["_caller"].isin(sel_callers)
        # Call Type
        if sel_ctypes and "All" not in sel_ctypes:
            mask = mask & dfc["_ctype"].isin(sel_ctypes)
        # Country Name
        if sel_countries and "All" not in sel_countries:
            mask = mask & dfc["_country"].isin(sel_countries)

        df_filtered = dfc.loc[mask].copy()

        # ----- KPIs -----
        total_secs = float(df_filtered["_secs"].sum())
        total_hms = _seconds_to_hms(total_secs)
        left, right = st.columns([1,3])
        with left:
            st.metric("Total Talk-time", total_hms)
            st.caption(f"Window: **{date_start} → {date_end}**, Time: **{start_time}–{end_time}**")
        with right:
            by_caller = (
                df_filtered.groupby("_caller")["_secs"]
                .sum()
                .reset_index()
                .rename(columns={"_caller":"Caller","_secs":"Talk time (sec)"})
                .sort_values("Talk time (sec)", ascending=False)
            )
            if not by_caller.empty:
                by_caller["Talk time (HH:MM:SS)"] = by_caller["Talk time (sec)"].map(_seconds_to_hms)
            st.dataframe(by_caller, use_container_width=True)

        # Optional: breakdowns
        with st.expander("Breakdowns (Call Type / Country)", expanded=False):
            cA, cB = st.columns(2)
            with cA:
                by_type = (
                    df_filtered.groupby("_ctype")["_secs"].sum().reset_index()
                    .rename(columns={"_ctype":"Call Type","_secs":"Talk time (sec)"})
                    .sort_values("Talk time (sec)", ascending=False)
                )
                if not by_type.empty:
                    by_type["Talk time (HH:MM:SS)"] = by_type["Talk time (sec)"].map(_seconds_to_hms)
                st.dataframe(by_type, use_container_width=True)
            with cB:
                by_country = (
                    df_filtered.groupby("_country")["_secs"].sum().reset_index()
                    .rename(columns={"_country":"Country Name","_secs":"Talk time (sec)"})
                    .sort_values("Talk time (sec)", ascending=False)
                )
                if not by_country.empty:
                    by_country["Talk time (HH:MM:SS)"] = by_country["Talk time (sec)"].map(_seconds_to_hms)
                st.dataframe(by_country, use_container_width=True)

        with st.expander("Preview filtered rows", expanded=False):
            show_cols = [c for c in calls_df.columns if not c.startswith("_")]
            st.dataframe(df_filtered[show_cols].head(500), use_container_width=True)

    else:
        st.info("Upload or point to your Calls CSV in the sidebar, then return here.")

# ======================
# Stubs for other views
# ======================
def _stub():
    st.info("This view is a placeholder in this consolidated file. Keep using your original app's implementation, or paste the corresponding block here.")

if view in {"MIS","Daily Business","Sales Tracker","AC Wise Detail","Leaderboard",
            "Funnel","Lead Movement","Stuck deals","Deal Velocity","Deal Decay","Carry Forward",
            "Predictibility","Business Projection","Buying Propensity","80-20","Trend & Analysis","Heatmap","Bubble Explorer","Master Graph",
            "Referrals","HubSpot Deal Score tracker","Marketing Lead Performance & Requirement"}:
    st.subheader(view)
    _stub()

# Footer
st.markdown('<div class="note">Tip: Replace any stubbed view with your original block to make this a single-file app.</div>', unsafe_allow_html=True)
