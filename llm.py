import json
import re

from openai import OpenAI

import config

_client = None


def _get_client():
    global _client
    if _client is None:
        if not config.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY is missing. Create a .env file and put your key in it.")
        _client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    return _client


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    resp = _get_client().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def chat_json(system: str, user: str) -> dict:
    raw = chat(system, user)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}
