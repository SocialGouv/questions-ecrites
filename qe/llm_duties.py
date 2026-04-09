"""Duty extraction helpers for parliamentary questions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from qe.clients.llm import SocleLLMClient


class DutyExtractor(Protocol):
    def request_duties(self, text: str) -> list[str]: ...


def normalize_json_content(content: str) -> str:
    if content.startswith("```") and content.endswith("```"):
        content = content.strip("`")
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3]
    return content.strip()


def parse_duties_payload(content: str) -> list[str]:  # noqa: C901
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM response is not valid JSON. Snippet: "
            f"{content[:200]}{'…' if len(content) > 200 else ''}"
        ) from exc

    duties_field = parsed.get("duties") or parsed.get("responsibilities")
    duties: list[str] = []

    def _clean_line(line: str) -> str:
        stripped = line.strip()
        stripped = stripped.lstrip("-•*0123456789.) ")
        return stripped.strip()

    if isinstance(duties_field, str):
        for line in duties_field.splitlines():
            cleaned = _clean_line(line)
            if cleaned:
                duties.append(cleaned)
    elif isinstance(duties_field, list):
        for item in duties_field:
            if isinstance(item, str):
                cleaned = _clean_line(item)
                if cleaned:
                    duties.append(cleaned)
            elif isinstance(item, dict):
                value = item.get("duty") or item.get("text") or item.get("description")
                if isinstance(value, str):
                    cleaned = _clean_line(value)
                    if cleaned:
                        duties.append(cleaned)

    if not duties:
        raise ValueError(
            "LLM response missing usable 'duties'. Parsed payload: "
            f"{json.dumps(parsed)[:300]}{'…' if len(json.dumps(parsed)) > 300 else ''}"
        )
    return duties


@dataclass(frozen=True)
class LLMQuestionDutyExtractor:
    client: SocleLLMClient

    def request_duties(self, text: str) -> list[str]:
        system_message = (
            "Vous êtes un analyste expert des questions parlementaires. À partir d'une question, "
            "identifiez les missions, obligations ou sujets clés que la question implique. Chaque devoir "
            "doit être exactement une phrase complète, détaillée, autonome et en français. "
            "Développez chaque acronyme lors de sa première apparition en écrivant « Forme longue (ACRONYME) ». "
            "Répondez uniquement avec du JSON strict, en respectant le schéma suivant : "
            '{"duties": ["devoir 1", "devoir 2", ...]}'
        )
        user_message = (
            "Question :\n{text}\n\n"
            "Règles :\n"
            "1. Produisez entre 3 et 10 devoirs selon le contenu.\n"
            "2. Chaque devoir doit décrire une seule action, obligation ou sujet clé.\n"
            "3. Les acronymes doivent être développés lorsqu'ils apparaissent.\n"
            "4. Écrivez en français.\n"
            "5. Fournissez uniquement du JSON, sans commentaire."
        ).format(text=text.strip())
        content = self.client.request_completion(
            system_message=system_message,
            user_message=user_message,
        )
        normalized = normalize_json_content(content)
        return parse_duties_payload(normalized)
