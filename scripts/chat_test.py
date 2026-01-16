#!/usr/bin/env python
"""Simple chat completion request to OpenWebUI Socle IA API."""

import os

import requests

BASE_URL = "https://pliage-prod.socle-ia.data-ia.prod.atlas.fabrique.social.gouv.fr"
MODEL = "mistralai/Ministral-3-8B-Instruct-2512"
# MODEL = "openai/gpt-oss-120b"


def chat_completion(prompt: str, model: str = MODEL) -> str:
    """Send a chat completion request to the Socle IA API."""
    api_key = os.environ.get("SOCLE_IA_API_KEY")
    if not api_key:
        raise ValueError("SOCLE_IA_API_KEY environment variable is not set")

    response = requests.post(
        f"{BASE_URL}/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    result = chat_completion("Say hello in French in one sentence.")
    print(result)
