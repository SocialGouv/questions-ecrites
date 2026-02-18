#!/usr/bin/env python
"""Simple chat completion test against the Socle IA API."""

from qe.clients.llm import SocleLLMClient
from qe.config import get_settings, require_api_key


def main() -> None:
    settings = get_settings()
    api_key = require_api_key("SOCLE_IA_API_KEY")

    client = SocleLLMClient(
        url=settings.chat_completions_url,
        model=settings.llm_model,
        api_key=api_key,
    )
    result = client.request_completion(
        system_message="You are a helpful assistant.",
        user_message="Say hello in French in one sentence.",
    )
    print(result)


if __name__ == "__main__":
    main()
