# --- Streamlit dashboard (main) ---
import streamlit as st
import pandas as pd
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
    cols_to_show = ["title", "summary", "severity", "possible_cause"]
    available_cols = [c for c in cols_to_show if c in final_df.columns]
    st.dataframe(final_df[available_cols], use_container_width=True)

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

    # --- Severity visual: line graph + cause origins ---
    st.markdown("---")
    st.header("📊 Severity Overview")

    # Normalize and numeric mapping
    final_df["possible_cause"] = final_df["possible_cause"].fillna("Unknown").astype(str)
    severity_map = {"Low": 1, "Medium": 2, "High": 3}
    final_df["severity_numeric"] = final_df["severity"].map(severity_map).fillna(2)

    # ---- Pie Chart for Severity Levels ----
    import matplotlib.pyplot as plt
    import numpy as np

    # Prepare data
    severity_counts = final_df["severity"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    labels = severity_counts.index.tolist()
    data = severity_counts.values

    # Custom colors: High=Red, Medium=Orange, Low=Green
    colors = ["#FF4C4C", "#FFA500", "#4CAF50"]  # red, orange, green

    # --- Function for labels (skip zeros) ---
    def autopct_format(pct, allvalues):
        total = np.sum(allvalues)
        count = int(round(pct / 100.0 * total))
        # Only show label if value > 0
        return f"{pct:.1f}%\n({count})" if count > 0 else ""

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        data,
        autopct=lambda pct: autopct_format(pct, data),
        explode=(0.05, 0.05, 0.05),
        labels=None,
        colors=colors,
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 10, 'color': 'black'}
    )

    # --- Equal aspect ratio ensures pie is circular ---
    ax.axis('equal')

    # --- Add legend & title ---
    ax.legend(
        wedges, 
        labels, 
        title="Severity Levels",
        loc="center left", 
        bbox_to_anchor=(1.25, 0, 0.5, 1),
        fontsize=10,
        title_fontsize=11
    )

    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title("Severity Distribution", fontsize=14, pad=20)

    # --- Adjust layout to avoid cutoff ---
    plt.tight_layout()

    # --- Display in Streamlit ---
    st.pyplot(fig)


    # Cause-origin inference
    st.subheader("Cause origins (one-line summary per issue)")

    def infer_origins(text: str):

        t = (text or "").lower()
        tags = []
        if any(k in t for k in ["stack", "trace", "exception", "traceback"]):
            tags.append("Runtime/Exception")
        if any(k in t for k in ["error", "bug", "wrong", "failed", "fail"]):
            tags.append("Bug/Error")
        if any(k in t for k in ["dependenc", "pip", "package", "module", "lib"]):
            tags.append("Dependency")
        if any(k in t for k in ["config", "env", "environment", "setting", "variable"]):
            tags.append("Configuration/Env")
        if any(k in t for k in ["timeout", "latency", "slow", "performance", "lag"]):
            tags.append("Performance")
        if any(k in t for k in ["auth", "token", "permission", "403", "401", "unauthor"]):
            tags.append("Auth/Permissions")
        if any(k in t for k in ["db", "database", "sql", "postgres", "mysql"]):
            tags.append("Database")
        if any(k in t for k in ["network", "dns", "tcp", "connection", "timeout"]):
            tags.append("Network")
        if any(k in t for k in ["ui", "css", "frontend", "react", "angular"]):
            tags.append("UI/Frontend")
        return list(dict.fromkeys(tags))[:4] or ["Unknown"]

    # One-line cause summary
    for i, row in final_df.head(30).iterrows():
        title = (row.get("title") or "").strip()
        desc = (row.get("description") or "").strip()
        inferred = infer_origins(title + " \n " + desc)
        provided = (row.get("possible_cause") or "").strip()
        hint = f"Provided cause: {provided}." if provided and provided.lower() != "unknown" else ""
        st.write(f"{i+1}. {title[:120]} — Causes: {', '.join(inferred)} {(' | ' + hint) if hint else ''}")

    # 💡 Recommended Fixes
    st.subheader("💡 Recommended Fixes")
    for i, row in final_df.head(30).iterrows():
        st.write(f"{i+1}. {row['title'][:120]} — 💡 {row.get('recommended_fix', 'N/A')}")

    # --- Download ---
    csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Summarized Issues", csv, f"{username}_{selected_repo}_summary.csv", "text/csv")
