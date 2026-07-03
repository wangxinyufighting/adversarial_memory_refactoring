from typing import Any, Dict, Iterable, List

from .models import SessionChunk


def _session_user_text(session: Iterable[Dict[str, Any]], include_assistant: bool = False) -> str:
    lines = []
    for turn in session:
        role = turn.get("role", "")
        if role == "user" or include_assistant:
            content = str(turn.get("content", "")).strip()
            if content:
                lines.append(content)
    return "\n".join(lines)


def entry_to_session_chunks(entry: Dict[str, Any], include_assistant: bool = False) -> List[SessionChunk]:
    """Convert one LongMemEval entry into session-level chunks.

    This follows UnifiedMem's core choice: each session becomes one chunk, and
    user messages are the default content because they carry user-memory facts.
    """
    case_id = entry.get("case_id") or entry.get("question_id") or entry.get("id") or "unknown_case"
    session_ids = entry.get("haystack_session_ids", [])
    sessions = entry.get("haystack_sessions", [])
    timestamps = entry.get("haystack_dates", [])

    chunks = []
    for order, session in enumerate(sessions):
        chunk_id = session_ids[order] if order < len(session_ids) else f"{case_id}_session_{order}"
        timestamp = timestamps[order] if order < len(timestamps) else ""
        chunks.append(
            SessionChunk(
                case_id=case_id,
                chunk_id=chunk_id,
                content=_session_user_text(session, include_assistant=include_assistant),
                timestamp=timestamp,
                order=order,
                metadata={"source": "longmemeval"},
            )
        )
    return chunks
