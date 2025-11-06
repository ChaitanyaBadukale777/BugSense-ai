# Streamlit dashboard (main)

# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from summarizer_local import summarize_issue
from embedder import EmbeddingIndex
from severity_model import predict_texts, load_model
from utils import fetch_issues_from_github
import requests
import time

st.set_page_config(page_title="BugSense AI - GitHub Issue Dashboard", layout="wide")
st.title("🚀 BugSense AI — GitHub Issue Intelligence Dashboard")

# --- Sidebar: GitHub Input ---
st.sidebar.header("🔧 GitHub Settings")

username = st.sidebar.text_input("GitHub Username (e.g. microsoft)", value="microsoft")
github_token = st.sidebar.text_input("GitHub Token (optional)", type="password")

# Fetch repositories dynamically
if username:
    st.sidebar.write("🔍 Fetching public repositories...")
    try:
        headers = {"Authorization": f"token {github_token}"} if github_token else {}
        repos_url = f"https://api.github.com/users/{username}/repos"
        repos = requests.get(repos_url, headers=headers).json()

        if isinstance(repos, list) and repos:
            repo_names = [r["name"] for r in repos]
            selected_repo = st.sidebar.selectbox("Select a repository", repo_names)
        else:
            selected_repo = None
            st.sidebar.warning("No repositories found or rate limit exceeded.")
    except Exception as e:
        selected_repo = None
        st.sidebar.error(f"Error fetching repositories: {e}")
else:
    selected_repo = None

max_issues = st.sidebar.slider("Max issues to fetch", min_value=5, max_value=50, value=10)
state = st.sidebar.selectbox("Issue state", ["open", "closed", "all"])
fetch_button = st.sidebar.button("Fetch & Analyze Issues")

# --- Main Process ---
if fetch_button and selected_repo:
    st.info(f"🔄 Fetching issues from `{username}/{selected_repo}` ...")
    try:
        df = fetch_issues_from_github(username, selected_repo, state, max_issues, token=github_token or None)
    except Exception as e:
        st.error(f"❌ Error fetching issues: {e}")
        st.stop()

    if df.empty:
        st.warning("No issues found for this repository.")
        st.stop()

    st.success(f"✅ Fetched {len(df)} issues successfully!")

    # --- Summarize with LLM (Ollama) ---
    summaries = []
    progress = st.progress(0)
    status = st.empty()
    for i, row in df.iterrows():
        title = row["title"]
        desc = row["description"]
        status.text(f"Analyzing issue {i+1}/{len(df)}: {title[:50]}")
        result = summarize_issue(title, desc)
        summaries.append(result)
        progress.progress((i + 1) / len(df))
        time.sleep(0.3)

    progress.empty()
    status.text("✅ Summarization complete!")

    result_df = pd.DataFrame(summaries)
    final_df = pd.concat([df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)

    st.subheader("📋 Summarized Issues")
    st.dataframe(final_df, use_container_width=True)

    # --- Duplicate Detection ---
    st.markdown("---")
    st.header("🧠 Duplicate / Similar Issues")
    embedder = EmbeddingIndex()
    embedder.build(final_df["description"].astype(str).tolist())
    dup_pairs = embedder.find_duplicates(threshold=0.78)
    if dup_pairs:
        st.success(f"Found {len(dup_pairs)} similar issue pairs:")
        for (i, j, sim) in dup_pairs:
            st.write(f"🔁 {i+1}: {final_df.loc[i, 'title']}  ↔  {j+1}: {final_df.loc[j, 'title']}  (Similarity: {sim:.2f})")
    else:
        st.info("No duplicates found above 0.78 similarity.")

    # --- Severity Visualizations (advanced) ---
    st.markdown("---")
    st.header("📊 Severity Dashboard")

    # Normalize fields
    final_df["possible_cause"] = final_df["possible_cause"].fillna("Unknown").astype(str)
    severity_map = {"Low": 1, "Medium": 2, "High": 3}
    final_df["severity_numeric"] = final_df["severity"].map(severity_map).fillna(2)

    # 1) Summary table with counts + percent
    severity_count = final_df["severity"].value_counts(dropna=False).rename_axis("Severity").reset_index(name="Count")
    total = severity_count["Count"].sum()
    severity_count["Percent"] = (severity_count["Count"] / total * 100).round(1)
    st.subheader("Severity counts & percentages")
    st.dataframe(severity_count.sort_values(by="Count", ascending=False), width='stretch')

    # 2) Pie chart (improved hover + hole)
    fig_pie = px.pie(severity_count, names="Severity", values="Count",
                     title="🧩 Severity Distribution", hole=0.4,
                     hover_data=["Percent"], labels={"Percent": "Percent (%)"})
    fig_pie.update_traces(textinfo='label+percent', hovertemplate='%{label}: %{value} issues (%{percent})')
    st.plotly_chart(fig_pie, width='stretch')

    # 3) Top possible causes (counts)
    cause_count = final_df["possible_cause"].value_counts().reset_index().head(12)
    cause_count.columns = ["Possible Cause", "Frequency"]
    fig_bar = px.bar(cause_count, x="Possible Cause", y="Frequency",
                     title="🧠 Top Recurring Root Causes", text_auto=True)
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, width='stretch')

    # 4) Stacked bar: top causes broken down by severity
    top_causes = final_df["possible_cause"].value_counts().head(8).index.tolist()
    breakdown = (final_df[final_df["possible_cause"].isin(top_causes)]
                 .groupby(["possible_cause", "severity"]).size().reset_index(name="count"))
    if not breakdown.empty:
        pivot = breakdown.pivot(index="possible_cause", columns="severity", values="count").fillna(0)
        pivot = pivot[[c for c in ["Low", "Medium", "High"] if c in pivot.columns]]
        fig_stack = px.bar(pivot, x=pivot.index, y=pivot.columns,
                           title="Stacked: Top Causes by Severity",
                           labels={"value": "Count", "possible_cause": "Cause"})
        fig_stack.update_layout(barmode='stack', xaxis_tickangle=-45)
        st.plotly_chart(fig_stack, width='stretch')

    # 5) Sunburst: severity -> cause -> count
    sunburst_df = final_df.groupby(["severity", "possible_cause"]).size().reset_index(name="count")
    if not sunburst_df.empty:
        fig_sun = px.sunburst(sunburst_df, path=["severity", "possible_cause"], values="count",
                              title="Severity → Possible Cause Breakdown")
        st.plotly_chart(fig_sun, width='stretch')

    # 6) Trend line with smoothing (rolling average over issue index)
    st.subheader("📈 Severity Trend (smoothed)")
    trend_df = final_df.reset_index().rename(columns={"index": "issue_index"})
    # ensure numeric severity exists
    if "severity_numeric" not in trend_df:
        trend_df["severity_numeric"] = trend_df["severity"].map(severity_map).fillna(2)
    # rolling mean (window depends on size)
    window = max(3, min(7, int(len(trend_df) / 4) or 3))
    trend_df["severity_roll"] = trend_df["severity_numeric"].rolling(window=window, min_periods=1, center=True).mean()
    fig_line = px.line(trend_df, x="issue_index", y=["severity_numeric", "severity_roll"],
                       labels={"value": "Severity (numeric)", "issue_index": "Issue index"},
                       title=f"Severity Trend (rolling window={window})")
    fig_line.update_yaxes(tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"]) 
    fig_line.update_traces(selector=dict(name='severity_numeric'), mode='markers+lines')
    st.plotly_chart(fig_line, width='stretch')

    # --- Download ---
    csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Summarized Issues", csv, f"{username}_{selected_repo}_summary.csv", "text/csv")
