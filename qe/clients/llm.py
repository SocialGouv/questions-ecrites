"""Socle IA chat completions API client."""

from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class SocleLLMClient:
    """Send chat completion requests to the Socle IA API."""

    url: str
    model: str
    api_key: str
    timeout: int = 120

    def request_completion(self, *, system_message: str, user_message: str) -> str:
        try:
            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "stream": False,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else "unknown"
            detail = exc.response.text.strip() if exc.response else ""
            snippet = detail[:500] + ("…" if len(detail) > 500 else "")
            raise RuntimeError(
                "LLM duty request failed "
                f"(status {status} from {self.url}). "
                "Response body snippet: "
                f"{snippet or '[empty response]'}"
            ) from exc
        return response.json()["choices"][0]["message"]["content"].strip()
