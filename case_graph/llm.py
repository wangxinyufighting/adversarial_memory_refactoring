import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from .models import EntityRecord, ExtractionResult, RelationshipRecord, SessionChunk
from .prompts import (
    DEFAULT_COMPLETION_DELIMITER,
    DEFAULT_RECORD_DELIMITER,
    DEFAULT_TUPLE_DELIMITER,
    build_extraction_user_prompt,
)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object, tolerating fenced responses and surrounding text."""
    text = _strip_code_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _clean_tuple_field(value: str) -> str:
    return value.strip().strip('"').strip("'").strip()


def parse_unifiedmem_extraction(text: str) -> ExtractionResult:
    """Parse UnifiedMem tuple-style graph extraction output."""
    text = _strip_code_fence(text)
    text = text.replace(DEFAULT_COMPLETION_DELIMITER, DEFAULT_RECORD_DELIMITER)
    entities = []
    relationships = []

    for raw_record in text.split(DEFAULT_RECORD_DELIMITER):
        match = re.search(r"\((.*)\)", raw_record.strip(), flags=re.DOTALL)
        if match is None:
            continue
        fields = [_clean_tuple_field(part) for part in match.group(1).split(DEFAULT_TUPLE_DELIMITER)]
        if not fields:
            continue
        record_type = fields[0].lower()
        if record_type == "entity" and len(fields) >= 4:
            entities.append(
                EntityRecord(
                    name=fields[1],
                    entity_type=fields[2] or "Other",
                    description=fields[3],
                )
            )
        elif record_type == "relationship" and len(fields) >= 5:
            try:
                weight = float(fields[4])
            except ValueError:
                weight = 1.0
            relationships.append(
                RelationshipRecord(
                    source=fields[1],
                    target=fields[2],
                    description=fields[3],
                    weight=weight,
                )
            )
    return ExtractionResult(entities=entities, relationships=relationships)


@dataclass
class OpenAIChatClient:
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 120
    thinking: Optional[Dict[str, str]] = None

    @classmethod
    def from_env(cls) -> "OpenAIChatClient":
        provider = os.environ.get("CASE_GRAPH_PROVIDER", "").strip().lower()
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        use_deepseek = provider == "deepseek" or (deepseek_key and not openai_key)

        if use_deepseek:
            api_key = deepseek_key
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is required when CASE_GRAPH_PROVIDER=deepseek.")
            return cls(
                model=os.environ.get("CASE_GRAPH_MODEL")
                or os.environ.get("DEEPSEEK_MODEL")
                or "deepseek-v4-flash",
                api_key=api_key,
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
                timeout=int(os.environ.get("CASE_GRAPH_TIMEOUT", "120")),
                thinking={"type": os.environ.get("DEEPSEEK_THINKING", "disabled")},
            )

        api_key = openai_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or DEEPSEEK_API_KEY is required for LLM extraction.")
        return cls(
            model=os.environ.get("CASE_GRAPH_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            timeout=int(os.environ.get("CASE_GRAPH_TIMEOUT", "120")),
        )

    def build_payload(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
        json_response: bool = True,
    ) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if json_response:
            payload["response_format"] = {"type": "json_object"}
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def _complete_content(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
        json_response: bool = True,
    ) -> str:
        payload = self.build_payload(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            json_response=json_response,
        )
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API request failed with HTTP {exc.code}: {body}") from exc
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]

    def complete_json(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        content = self._complete_content(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            json_response=True,
        )
        return parse_json_object(content)

    def complete_text(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self._complete_content(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            json_response=False,
        )


class TextChatClient(Protocol):
    def complete_text(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        ...


@dataclass
class LLMExtractor:
    client: Optional[TextChatClient] = None
    max_input_chars: int = 12000
    max_output_tokens: int = 1200

    def __post_init__(self) -> None:
        self.max_input_chars = int(os.environ.get("CASE_GRAPH_MAX_INPUT_CHARS", self.max_input_chars))
        self.max_output_tokens = int(os.environ.get("CASE_GRAPH_MAX_OUTPUT_TOKENS", self.max_output_tokens))

    def extract(self, chunk: SessionChunk) -> ExtractionResult:
        content = self._truncate_content(chunk.content)
        raw_output = self._client().complete_text(
            system_prompt=None,
            user_prompt=build_extraction_user_prompt(content, chunk.timestamp),
            max_tokens=self.max_output_tokens,
        )
        return parse_unifiedmem_extraction(raw_output)

    def _client(self) -> TextChatClient:
        if self.client is None:
            self.client = OpenAIChatClient.from_env()
        return self.client

    def _truncate_content(self, content: str) -> str:
        if self.max_input_chars <= 0 or len(content) <= self.max_input_chars:
            return content
        return content[: self.max_input_chars]
