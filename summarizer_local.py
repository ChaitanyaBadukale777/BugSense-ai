# summarizer_local.py
"""
Summarizer that calls a local Ollama model via subprocess.
Make sure ollama is installed and a model is pulled, e.g. `ollama pull mistral`.
"""

import subprocess
import json
from typing import Dict

OLLAMA_MODEL = "mistral"  # change if you use a different local model


def _build_prompt(title: str, body: str) -> str:
    prompt = f"""
    You are an expert QA engineer. Analyze the GitHub issue below and return valid JSON with the following keys (no extra text, no markdown, no emojis):

    - summary: (short, 2-line overview of the issue)
    - severity: (Low / Medium / High)
    - possible_cause: (concise 2-line explanation of what caused the issue)
    - recommended_fix: (a multi-line numbered list of at least 5 concise, one-line actionable steps — each strictly starting with "1)", "2)", "3)", etc. on separate lines)

    STRICT FORMATTING RULES:
    - Return strictly valid JSON (no markdown, no commentary).
    - recommended_fix must be a **single string**, not an array.
    - Each fix must start with a number followed by a closing parenthesis (e.g., "1) " not "1." or "1 -").
    - Each fix must be on its own line, separated by newline characters (\n).
    - No emojis, bullets, dashes, or decorative symbols allowed.
    - Keep sentences short, factual, and human-readable.
    - If unsure of severity, mark as "Medium".

    Issue Title: {title}
    Issue Body: {body}
    """
    return prompt.strip()



def summarize_issue(title: str, body: str, timeout: int = 60) -> Dict[str, str]:
    prompt = _build_prompt(title, body)

    try:
        command = ["ollama", "run", OLLAMA_MODEL, prompt]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        output = result.stdout.strip()
    except Exception as e:
        return {
            "summary": f"Error running ollama: {e}",
            "severity": "Medium",
            "possible_cause": "LLM error",
            "recommended_fix": "N/A"
        }

    try:
        parsed = json.loads(output)

        # Handle recommended_fix safely (string or list)
        fix_value = parsed.get("recommended_fix", "N/A")
        if isinstance(fix_value, list):
            # Join list items into multiline string
            fix_value = "\n".join(str(item).strip() for item in fix_value)
        elif not isinstance(fix_value, str):
            # Convert any non-string to string
            fix_value = str(fix_value)
        fix_value = fix_value.strip()

        return {
            "summary": parsed.get("summary", "").strip(),
            "severity": parsed.get("severity", "Medium").strip(),
            "possible_cause": parsed.get("possible_cause", "").strip(),
            "recommended_fix": fix_value
        }

    except json.JSONDecodeError:
        # When model returns invalid JSON
        return {
            "summary": output[:300],
            "severity": "Medium",
            "possible_cause": "Unclear",
            "recommended_fix": "N/A"
        }
