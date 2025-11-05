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
You are an expert QA engineer. Analyze the GitHub issue below and return ONLY valid JSON with the keys:
  - summary (short, one-line)
  - severity (Low / Medium / High)
  - possible_cause (one short line)
Make severity conservative: if unsure, mark Medium.

Issue Title: {title}
Issue Body: {body}
"""
    return prompt.strip()


def summarize_issue(title: str, body: str, timeout: int = 60) -> Dict[str, str]:
    prompt = _build_prompt(title, body)

    try:
        # call ollama run <model> "<prompt>"
        # we wrap the prompt as a single argument
        command = ["ollama", "run", OLLAMA_MODEL, prompt]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        output = result.stdout.strip()
    except Exception as e:
        return {"summary": f"Error running ollama: {e}", "severity": "Medium", "possible_cause": "LLM error"}

    # Try parse JSON
    try:
        parsed = json.loads(output)
        # ensure keys exist
        return {
            "summary": parsed.get("summary", "").strip(),
            "severity": parsed.get("severity", "Medium").strip(),
            "possible_cause": parsed.get("possible_cause", "").strip()
        }
    except json.JSONDecodeError:
        # If model didn't return pure JSON, try to salvage common patterns:
        # Very simple fallback: return full output as summary
        return {"summary": output[:300], "severity": "Medium", "possible_cause": "Unclear"}
