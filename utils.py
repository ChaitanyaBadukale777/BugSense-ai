# utils.py

""" 
Helper utilities:
- fetch_issues_from_github(owner, repo, state, max_issues)
- parse_pdf_text(file_like)

"""

import requests
import pandas as pd
import io
from PyPDF2 import PdfReader

GITHUB_API_URL = "https://api.github.com"

def fetch_issues_from_github(owner: str, repo: str, state: str = "open", max_issues: int = 20, token: str = None):
    issues = []
    per_page = 50 if max_issues > 50 else max_issues
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/issues"
    params = {"state": state, "per_page": per_page}
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    # filter out pull requests
    for item in data:
        if "pull_request" in item:
            continue
        issues.append({
            "id": item.get("id"),
            "number": item.get("number"),
            "title": item.get("title"),
            "description": item.get("body") or ""
        })
        if len(issues) >= max_issues:
            break
    return pd.DataFrame(issues)


def parse_pdf_text(file_like) -> str:
    reader = PdfReader(file_like)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)