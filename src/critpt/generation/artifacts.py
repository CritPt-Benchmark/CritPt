"""
Helpers for persisting inspect.ai task states alongside the cleaned submissions.

The original solver mixed prompt orchestration with several utility functions that
serialize task states into JSON, capture prompts, and read cached completions. In
the refactor we keep those responsibilities in a dedicated module so the solver
can focus on the high-level generation flow (system prompt -> user prompt -> parse
template).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import ChatMessage, ChatMessageTool
from inspect_ai.solver import TaskState


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion so inspect TaskState objects can be dumped to JSON."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump())
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)
    return str(value)


def _normalize_content_type(content_type: str) -> str:
    if content_type == "text":
        return "content"
    if content_type == "tool_output":
        return "tool_call"
    return content_type


def read_chat_message_content(chat_message: ChatMessage) -> List[Dict[str, str]]:
    """Inspect message content blocks and normalize them for serialization."""
    if isinstance(chat_message.content, str):
        if not isinstance(chat_message, ChatMessageTool):
            return [{"type": "text", "content": chat_message.content}]

        output = chat_message.content.strip()
        fenced = output if output.startswith("```") else f"```\n{output}\n```"
        return [{"type": "tool_output", "content": fenced}]

    ret = []
    for content in chat_message.content:
        if isinstance(content, ContentText):
            ret.append({"type": "text", "content": content.text})
        elif isinstance(content, ContentReasoning):
            ret.append({"type": "reasoning", "content": content.reasoning})
        else:
            ret.append({"type": "unknown", "content": str(content)})

    if chat_message.tool_calls is not None:
        for tool_call in chat_message.tool_calls:
            if tool_call.view is not None:
                content = tool_call.view.content
                if not content.strip().startswith("```"):
                    content = f"```\n{content}\n```"
                ret.append({"type": "tool_call", "content": content})
            else:
                ret.append({"type": "tool_call", "content": "None"})

    return ret


def _dedupe_raw_payload(raw_payload: Any, role: str) -> Any:
    """
    The normalized `contents` list already exposes the user/system text, so drop
    duplicated role/content keys from the raw payload while preserving other
    inspect metadata (tool calls, ids, usage, etc.).
    """
    if not isinstance(raw_payload, dict):
        return raw_payload

    raw_role = raw_payload.get("role")
    if isinstance(raw_role, str) and raw_role.lower() == role:
        raw_payload.pop("role", None)

    raw_payload.pop("content", None)

    return raw_payload or None


def _serialize_chat_message(message: ChatMessage) -> Dict[str, Any]:
    role = getattr(message, "role", type(message).__name__).lower()
    contents = []
    for block in read_chat_message_content(message):
        contents.append(
            {
                "content_type": _normalize_content_type(block.get("type", "content")),
                "content_text": block.get("content", ""),
            }
        )

    serialized: Dict[str, Any] = {
        "role": role,
        "contents": contents,
    }

    raw_payload = None
    if hasattr(message, "model_dump"):
        try:
            raw_payload = _to_jsonable(message.model_dump())
        except Exception:  # pragma: no cover - defensive fallback
            raw_payload = {"content": _to_jsonable(getattr(message, "content", None))}
    else:  # pragma: no cover - defensive fallback
        raw_payload = {"content": _to_jsonable(getattr(message, "content", None))}

    raw_payload = _dedupe_raw_payload(raw_payload, role)
    if raw_payload:
        serialized["raw"] = raw_payload

    return serialized


def serialize_task_state(state: TaskState | None) -> Dict[str, Any] | None:
    """Serialize a TaskState into a JSON friendly dict."""
    if state is None:
        return None

    serialized: Dict[str, Any] = {
        "messages": [_serialize_chat_message(message) for message in state.messages],
    }

    if getattr(state, "output", None):
        try:
            serialized["output"] = _to_jsonable(state.output.model_dump())
        except Exception:  # pragma: no cover - defensive fallback
            serialized["output"] = _to_jsonable(state.output)

    if getattr(state, "metadata", None):
        serialized["metadata"] = _to_jsonable(state.metadata)

    if getattr(state, "tools", None):
        serialized["tools"] = _to_jsonable(state.tools)

    if getattr(state, "tags", None):
        serialized["tags"] = list(state.tags)

    return serialized


def build_step_payload(
    step: str,
    processed_state: TaskState | None,
    parsing_state: TaskState | None,
    prompt: str | None,
) -> Dict[str, Any]:
    """Pack everything we want to archive for a single generation step."""
    processed_serialized = serialize_task_state(processed_state)
    parsing_serialized = serialize_task_state(parsing_state)

    chat_messages = []
    if parsing_serialized and parsing_serialized.get("messages"):
        chat_messages = parsing_serialized["messages"]
    elif processed_serialized and processed_serialized.get("messages"):
        chat_messages = processed_serialized["messages"]

    output_summary = None
    if parsing_serialized and isinstance(parsing_serialized.get("output"), dict):
        output_summary = parsing_serialized["output"]
    elif processed_serialized and isinstance(processed_serialized.get("output"), dict):
        output_summary = processed_serialized["output"]

    summary_block: Dict[str, Any] = {}
    if output_summary is not None:
        if "completion" in output_summary:
            summary_block["completion"] = output_summary.get("completion")
        if "finish_reason" in output_summary:
            summary_block["finish_reason"] = output_summary.get("finish_reason")
        usage_summary = output_summary.get("usage")
        if usage_summary is not None:
            summary_block["usage"] = usage_summary

    payload: Dict[str, Any] = {
        "artifact_type": "step",
        "version": 2,
        "step": step,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "chat": chat_messages,
        "summary": summary_block or None,
        "inspect": {
            "processed_state": processed_serialized,
            "parsing_state": parsing_serialized,
        },
    }

    if output_summary is not None:
        payload["inspect"]["output"] = output_summary

    return payload


def write_step_artifact(path: Path, payload: Dict[str, Any]) -> None:
    """Persist artifacts/main.json style files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_completion_from_artifact(path: Path) -> Optional[str]:
    """Best-effort attempt to recover the assistant completion from an artifact file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # v2 payloads: completion lives inside the summary block
    summary = payload.get("summary")
    if isinstance(summary, dict):
        completion = summary.get("completion")
        if isinstance(completion, str):
            return completion

    inspect_block = payload.get("inspect")
    if isinstance(inspect_block, dict):
        output = inspect_block.get("output")
        if isinstance(output, dict):
            completion = output.get("completion")
            if isinstance(completion, str):
                return completion

        parsing_output = (
            inspect_block.get("parsing_state", {}).get("output")
            if isinstance(inspect_block.get("parsing_state"), dict)
            else None
        )
        if isinstance(parsing_output, dict):
            completion = parsing_output.get("completion")
            if isinstance(completion, str):
                return completion

        processed_output = (
            inspect_block.get("processed_state", {}).get("output")
            if isinstance(inspect_block.get("processed_state"), dict)
            else None
        )
        if isinstance(processed_output, dict):
            completion = processed_output.get("completion")
            if isinstance(completion, str):
                return completion

    # Backward compatibility: legacy payloads stored these at the top level
    output = payload.get("output")
    if isinstance(output, dict):
        completion = output.get("completion")
        if isinstance(completion, str):
            return completion

    parsing_output = (
        payload.get("parsing_state", {}).get("output")
        if isinstance(payload.get("parsing_state"), dict)
        else None
    )
    if isinstance(parsing_output, dict):
        completion = parsing_output.get("completion")
        if isinstance(completion, str):
            return completion

    processed_output = (
        payload.get("processed_state", {}).get("output")
        if isinstance(payload.get("processed_state"), dict)
        else None
    )
    if isinstance(processed_output, dict):
        completion = processed_output.get("completion")
        if isinstance(completion, str):
            return completion

    return None


def load_completion_from_submission(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            submission_data = json.load(f)
            completion = submission_data.get("generated_code")
            if isinstance(completion, str):
                return completion
    except Exception:
        return None
    return None


def load_cached_completion(
    output_dir: Path | None,
    step_name: str,
    problem_id: Optional[str],
) -> Optional[str]:
    """
    Try to recover a completion for a step from the artifact, legacy .out files, or
    the cleaned submission JSON.
    """
    if output_dir is None:
        return None

    artifact_path = output_dir / f"{step_name}.json"
    legacy_path = output_dir / f"{step_name}.out"

    completion = None
    if artifact_path.exists():
        completion = load_completion_from_artifact(artifact_path)
    elif legacy_path.exists():
        completion = legacy_path.read_text(encoding="utf-8")

    if completion is not None:
        return completion

    if problem_id:
        submission_path = output_dir / f"{problem_id}.json"
        completion = load_completion_from_submission(submission_path)

    return completion


def save_step_artifact(
    output_dir: Path | None,
    step: str,
    processed_state: TaskState | None,
    parsing_state: TaskState | None,
    prompt: str | None,
) -> None:
    """High-level helper invoked by the solver after each successful step."""
    if output_dir is None:
        return

    payload = build_step_payload(step, processed_state, parsing_state, prompt)
    artifact_path = output_dir / f"{step}.json"
    write_step_artifact(artifact_path, payload)
