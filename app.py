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

    # --- Severity Visualizations ---
    st.markdown("---")
    st.header("📊 Severity Dashboard")

    severity_count = final_df["severity"].value_counts().reset_index()
    severity_count.columns = ["Severity", "Count"]
    fig_pie = px.pie(severity_count, names="Severity", values="Count", title="🧩 Severity Distribution", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

    cause_count = final_df["possible_cause"].value_counts().reset_index().head(10)
    cause_count.columns = ["Possible Cause", "Frequency"]
    fig_bar = px.bar(cause_count, x="Possible Cause", y="Frequency",
                     title="🧠 Top Recurring Root Causes", text_auto=True)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Trend Line ---
    severity_map = {"Low": 1, "Medium": 2, "High": 3}
    final_df["severity_numeric"] = final_df["severity"].map(severity_map).fillna(2)
    fig_line = px.line(final_df.reset_index(), x="index", y="severity_numeric",
                       title="📈 Severity Trend (By Issue Index)", markers=True)
    fig_line.update_yaxes(tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"])
    st.plotly_chart(fig_line, use_container_width=True)

    # --- Download ---
    csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Summarized Issues", csv, f"{username}_{selected_repo}_summary.csv", "text/csv")
