
# ---- Compact Status Bar (badges) ----

def _render_status_bar(badges):
    import streamlit as st
    def pill(text, color="#e5e7eb"):
        return f'<span class="legend-pill" style="background:{color}">{text}</span>'
    st.markdown("<div>" + "".join(pill(t,c) for t,c in badges) + "</div>", unsafe_allow_html=True)

# ---- Sidebar nav (master → sub/pills) ----
import streamlit as st

with st.sidebar:
    st.markdown("### Navigation")
    MASTER_SECTIONS = {
        "Performance": ["Cash-in","Dashboard","MIS","Daily Business","Sales Tracker","AC Wise Detail","Leaderboard","Call Talk-time Report"],
        "Funnel & Movement": ["Funnel","Lead Movement","Stuck deals","Deal Velocity","Deal Decay","Carry Forward"],
        "Insights & Forecast": ["Predictibility","Business Projection","Buying Propensity","80-20","Trend & Analysis","Heatmap","Bubble Explorer","Master Graph"],
        "Marketing": ["Referrals","HubSpot Deal Score tracker","Marketing Lead Performance & Requirement"],
    }
    master = st.radio("Sections", list(MASTER_SECTIONS.keys()), index=0, key="nav_master")
    sub_views = MASTER_SECTIONS.get(master, [])
    if 'nav_sub' not in st.session_state or st.session_state.get('nav_master_prev') != master:
        st.session_state['nav_sub'] = sub_views[0] if sub_views else ''
    st.session_state['nav_master_prev'] = master
    sub = st.session_state['nav_sub']

# ---- Top legend ----
legend_labels = ["Valid", "Invalid", "Carry Forward"]
legend_colors = ["#d1fae5", "#fee2e2", "#e5e7eb"]
pill_map = {l: f'<span class="legend-pill" style="background:{c}">{l}</span>' for l,c in zip(legend_labels, legend_colors)}

# ---- Active view state ----
try:
    _master = master if 'master' in locals() else st.session_state.get('nav_master', '')
    _view = st.session_state.get('nav_sub', locals().get('view', ''))
    if not _view and 'MASTER_SECTIONS' in globals():
        _cands = MASTER_SECTIONS.get(_master, [])
        _view = _cands[0] if _cands else ''
    cur_sub = _view
    cols = st.columns(len(MASTER_SECTIONS.get(_master, [])) or 1)
    for i, v in enumerate(MASTER_SECTIONS.get(_master, [])):
        if cols[i].button(v, use_container_width=True):
            st.session_state['nav_sub'] = v
            cur_sub = v
            st.rerun()
    view = st.session_state.get('nav_sub', cur_sub)
except Exception:
    view = locals().get('view', 'MIS')

# ---- Existing views (MIS, Leaderboard, etc.) remain unchanged ----

# ===========================
# 📞 Performance ▸ Call Talk-time Report
# ===========================
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, time, date

def _ctt_parse_hms_to_seconds(x: str) -> int:
    if pd.isna(x):
        return 0
    x = str(x).strip()
    try:
        parts = x.split(":")
        if len(parts) != 3:
            return 0
        hh, mm, ss = parts
        return int(hh) * 3600 + int(mm) * 60 + int(ss)
    except Exception:
        return 0

def _ctt_seconds_to_hms(total_seconds: int) -> str:
    total_seconds = int(total_seconds or 0)
    hh = total_seconds // 3600
    rem = total_seconds % 3600
    mm = rem // 60
    ss = rem % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def _ctt_ensure_datetime(df, date_col="Date", time_col="Time"):
    d = pd.to_datetime(df[date_col].astype(str), errors="coerce", dayfirst=False)
    t_try = pd.to_datetime(df[time_col].astype(str), errors="coerce").dt.time
    if t_try.isna().any():
        def _fix_t(val):
            try:
                p = str(val).split(":")
                if len(p) == 2:
                    h, m = p; s = 0
                elif len(p) == 3:
                    h, m, s = p
                else:
                    return None
                return time(int(h), int(m), int(s))
            except Exception:
                return None
        t_try = df[time_col].astype(str).map(_fix_t)
    dt = pd.to_datetime(d.dt.strftime("%Y-%m-%d") + " " + pd.Series(t_try).astype(str), errors="coerce")
    df = df.assign(_dt=dt).dropna(subset=["_dt"])
    return df

def _ctt_filter_window(df, start_date, end_date, start_time, end_time):
    mask_date = df["_dt"].dt.date.between(start_date, end_date)
    t = df["_dt"].dt.time
    if start_time <= end_time:
        mask_time = (t >= start_time) & (t <= end_time)
    else:
        mask_time = (t >= start_time) | (t <= end_time)
    return df[mask_date & mask_time]

def _ctt_add_duration_seconds(df, duration_col="Call Duration"):
    if duration_col not in df.columns:
        for c in df.columns:
            if "duration" in c.lower() or "talk" in c.lower():
                duration_col = c
                break
    secs = df[duration_col].apply(_ctt_parse_hms_to_seconds)
    return df.assign(_secs=secs)

def _ctt_download_csv_button(label, df, file_name):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=file_name, mime="text/csv")

def _ctt_pick(df, preferred, cands):
    if preferred and preferred in df.columns: return preferred
    for c in cands:
        if c in df.columns: return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    return None

def _render_call_talktime_report():
    st.subheader("Performance — Call Talk-time Report")
    st.caption("Upload the call activity CSV (columns: Date, Time, Caller, Country Name, Call Type, Call Duration as HH:MM:SS).")

    upl = st.file_uploader("Upload activity feed CSV", type=["csv"], key="ctt_upl")
    if upl is None:
        st.info("Please upload the activityFeedReport_*.csv to continue.")
        return
    try:
        df_raw = pd.read_csv(upl)
    except Exception:
        text = upl.read().decode("utf-8", errors="ignore")
        df_raw = pd.read_csv(StringIO(text))

    date_col     = _ctt_pick(df_raw, None, ["Date","Call Date"])
    time_col     = _ctt_pick(df_raw, None, ["Time","Call Time"])
    caller_col   = _ctt_pick(df_raw, None, ["Caller","Agent","Counsellor","Counselor","Caller Name","User"])
    type_col     = _ctt_pick(df_raw, None, ["Call Type","Type"])
    country_col  = _ctt_pick(df_raw, None, ["Country Name","Country"])
    duration_col = _ctt_pick(df_raw, None, ["Call Duration","Duration","Talk Time"])

    needed = [date_col, time_col, caller_col, duration_col]
    if any(x is None for x in needed):
        st.error("Missing required columns. Need Date, Time, Caller, Call Duration.")
        return

    df = df_raw.copy()
    df = _ctt_ensure_datetime(df, date_col=date_col, time_col=time_col)
    if df.empty:
        st.warning("No valid rows after parsing Date & Time.")
        return

    df = _ctt_add_duration_seconds(df, duration_col=duration_col)
    if "_secs" not in df.columns:
        st.error("Could not compute duration seconds from Call Duration.")
        return

    type_col = type_col if (type_col and type_col in df.columns) else None
    country_col = country_col if (country_col and country_col in df.columns) else None

    min_d = df["_dt"].dt.date.min()
    max_d = df["_dt"].dt.date.max()
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        d_start = st.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d, key="ctt_start_d")
    with c2:
        d_end = st.date_input("End date", value=max_d, min_value=min_d, max_value=max_d, key="ctt_end_d")
    with c3:
        t_start = st.time_input("Start time", value=time(0,0,0), key="ctt_start_t")
    with c4:
        t_end = st.time_input("End time", value=time(23,59,59), key="ctt_end_t")

    df_f = _ctt_filter_window(df, d_start, d_end, t_start, t_end)

    callers = sorted(df_f[caller_col].dropna().astype(str).unique().tolist())
    types = sorted(df_f[type_col].dropna().astype(str).unique().tolist()) if type_col else []
    countries = sorted(df_f[country_col].dropna().astype(str).unique().tolist()) if country_col else []

    cA, cB, cC = st.columns([1,1,1])
    with cA:
        sel_callers = st.multiselect("Caller(s)", callers, default=callers, key="ctt_callers")
    with cB:
        sel_types = st.multiselect("Call Type(s)", types, default=types, key="ctt_types") if types else []
    with cC:
        sel_countries = st.multiselect("Country Name(s)", countries, default=countries, key="ctt_ctys") if countries else []

    mask = df_f[caller_col].astype(str).isin(sel_callers)
    if type_col and sel_types:
        mask &= df_f[type_col].astype(str).isin(sel_types)
    if country_col and sel_countries:
        mask &= df_f[country_col].astype(str).isin(sel_countries)
    df_f = df_f[mask].copy()
    if df_f.empty:
        st.warning("No rows after applying filters.")
        return

    total_secs = int(df_f["_secs"].sum())
    total_calls = int(len(df_f))
    avg_secs = int(round(df_f["_secs"].mean())) if total_calls else 0
    med_secs = int(df_f["_secs"].median()) if total_calls else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Talk Time", _ctt_seconds_to_hms(total_secs))
    m2.metric("# Calls", f"{total_calls:,}")
    m3.metric("Avg Call Duration", _ctt_seconds_to_hms(avg_secs))
    m4.metric("Median Call Duration", _ctt_seconds_to_hms(med_secs))

    # Caller-wise totals
    caller_tot = (df_f.groupby(caller_col, dropna=False)["_secs"].sum().reset_index()
                  .rename(columns={"_secs": "Total Seconds"}).sort_values("Total Seconds", ascending=False))
    caller_tot["Total Talk Time"] = caller_tot["Total Seconds"].map(_ctt_seconds_to_hms)

    gt60 = df_f[df_f["_secs"] > 60]
    caller_tot_gt60 = (gt60.groupby(caller_col, dropna=False)["_secs"].sum().reset_index()
                       .rename(columns={"_secs": "Total Seconds (>60s)"}).sort_values("Total Seconds (>60s)", ascending=False))
    caller_tot_gt60["Total Talk Time (>60s)"] = caller_tot_gt60["Total Seconds (>60s)"].map(_ctt_seconds_to_hms)

    if country_col:
        country_tot = (df_f.groupby(country_col, dropna=False)["_secs"].sum().reset_index()
                       .rename(columns={"_secs": "Total Seconds"}).sort_values("Total Seconds", ascending=False))
        country_tot["Total Talk Time"] = country_tot["Total Seconds"].map(_ctt_seconds_to_hms)
    else:
        country_tot = pd.DataFrame(columns=["Country", "Total Seconds", "Total Talk Time"])

    if country_col:
        country_tot_gt60 = (gt60.groupby(country_col, dropna=False)["_secs"].sum().reset_index()
                            .rename(columns={"_secs": "Total Seconds (>60s)"}).sort_values("Total Seconds (>60s)", ascending=False))
        country_tot_gt60["Total Talk Time (>60s)"] = country_tot_gt60["Total Seconds (>60s)"].map(_ctt_seconds_to_hms)
    else:
        country_tot_gt60 = pd.DataFrame(columns=["Country", "Total Seconds (>60s)", "Total Talk Time (>60s)"])

    st.markdown("### 1) Caller wise — Total Call Duration")
    st.dataframe(caller_tot[[caller_col, "Total Talk Time", "Total Seconds"]], use_container_width=True)
    _ctt_download_csv_button("⬇️ Download Caller Totals (All)", caller_tot, "caller_total_all.csv")

    st.markdown("### 2) Caller wise — Total Call Duration (> 60 sec)")
    st.dataframe(caller_tot_gt60[[caller_col, "Total Talk Time (>60s)", "Total Seconds (>60s)"]], use_container_width=True)
    _ctt_download_csv_button("⬇️ Download Caller Totals (>60s)", caller_tot_gt60, "caller_total_gt60.csv")

    if country_col:
        st.markdown("### 3) Country wise — Total Call Duration")
        st.dataframe(country_tot[[country_col, "Total Talk Time", "Total Seconds"]], use_container_width=True)
        _ctt_download_csv_button("⬇️ Download Country Totals (All)", country_tot, "country_total_all.csv")

        st.markdown("### 4) Country wise — Total Call Duration (> 60 sec)")
        st.dataframe(country_tot_gt60[[country_col, "Total Talk Time (>60s)", "Total Seconds (>60s)"]], use_container_width=True)
        _ctt_download_csv_button("⬇️ Download Country Totals (>60s)", country_tot_gt60, "country_total_gt60.csv")
    else:
        st.info("Country column not found — skipping country-wise breakdowns.")

    st.markdown("### 5) Caller-wise Calling Journey (Hour-of-Day)")
    df_f = df_f.assign(_hour=df_f["_dt"].dt.hour)
    per_caller_hour = (df_f.groupby([caller_col, "_hour"], dropna=False)["_secs"].sum().reset_index()
                       .rename(columns={"_secs": "Total Seconds"}))

    def _agg_max_min(g):
        if g.empty:
            return pd.Series({"Max Hour": np.nan, "Max Hour Talk Time": 0, "Min Hour": np.nan, "Min Hour Talk Time": 0})
        g = g.sort_values("Total Seconds", ascending=False)
        max_hour = int(g.iloc[0]["_hour"]); max_val  = int(g.iloc[0]["Total Seconds"])
        g_nonzero = g[g["Total Seconds"] > 0]
        g_min = (g_nonzero if not g_nonzero.empty else g).sort_values("Total Seconds", ascending=True).iloc[0]
        min_hour = int(g_min["_hour"]); min_val = int(g_min["Total Seconds"])
        return pd.Series({"Max Hour": max_hour, "Max Hour Talk Time": max_val, "Min Hour": min_hour, "Min Hour Talk Time": min_val})

    caller_hour_summary = per_caller_hour.groupby(caller_col, dropna=False).apply(_agg_max_min).reset_index()
    caller_hour_summary["Max Hour Talk Time (HH:MM:SS)"] = caller_hour_summary["Max Hour Talk Time"].map(_ctt_seconds_to_hms)
    caller_hour_summary["Min Hour Talk Time (HH:MM:SS)"] = caller_hour_summary["Min Hour Talk Time"].map(_ctt_seconds_to_hms)

    st.markdown("**Max/Min Hour per Caller (by Talk-time)**")
    st.dataframe(caller_hour_summary[[caller_col, "Max Hour", "Max Hour Talk Time (HH:MM:SS)", "Min Hour", "Min Hour Talk Time (HH:MM:SS)"]], use_container_width=True)
    _ctt_download_csv_button("⬇️ Download Caller Hour Summary", caller_hour_summary, "caller_hour_summary.csv")

    f_sel = st.selectbox("Focus: Caller hour profile", ["(All)"] + callers, index=0, key="ctt_focus")
    if f_sel != "(All)":
        foc = per_caller_hour[per_caller_hour[caller_col] == f_sel].sort_values("_hour")
        foc["Talk Time (HH:MM:SS)"] = foc["Total Seconds"].map(_ctt_seconds_to_hms)
        st.markdown(f"**Hourly Distribution for `{f_sel}`**")
        st.dataframe(foc[["_hour", "Talk Time (HH:MM:SS)", "Total Seconds"]], use_container_width=True)
        _ctt_download_csv_button(f"⬇️ Download Hourly Profile — {f_sel}", foc, f"hourly_profile_{f_sel}.csv")

try:
    if 'view' in globals() and view == "Call Talk-time Report":
        _render_call_talktime_report()
except Exception:
    pass
