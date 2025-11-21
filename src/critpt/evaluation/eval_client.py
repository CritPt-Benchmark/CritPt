"""
HTTP clients for talking to the evaluation server.

The server now exposes a single `/evaluate` endpoint that accepts a batch of
submissions.  PUBLIC/PRIVACY modes respond with aggregate metrics
(`{"accuracy": ..., "timeout_rate": ...}`) while DEBUG mode returns the full
per-example results plus the same aggregate metrics.  The helpers below smooth
over these details so scripts can continue to process submissions easily.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import json
import httpx
from httpx import Limits

from critpt.submission import Submission


def _notify_progress(callback: Optional[Callable[[int, int], None]], completed: int, total: int) -> None:
    if callback is None:
        return
    try:
        callback(completed, total)
    except Exception:
        # Swallow progress callback errors so they do not break evaluation
        pass


def _serialize_submission(submission: Submission) -> Dict[str, Any]:
    """Convert a Submission dataclass into the wire format."""
    return {
        "problem_id": submission.problem_id,
        "generated_code": submission.generated_code,
        "model": submission.model,
        "generation_config": submission.generation_config,
        "messages": submission.messages,
    }


def _build_batch_payload(
    submissions: List[Submission],
    batch_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the JSON payload expected by /evaluate."""
    return {
        "submissions": [_serialize_submission(sub) for sub in submissions],
        "batch_metadata": batch_metadata or {},
    }


def _extract_single_result(batch_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull the first evaluation result out of a DEBUG response.

    Raises:
        ValueError: If the server returned aggregate metrics instead of details.
    """
    results = batch_response.get("results")

    if isinstance(results, list) and results:
        result = results[0]
    elif isinstance(results, dict) and results:
        # Legacy dict format; grab the first item.
        first_key = next(iter(results))
        result = results[first_key]
    else:
        raise ValueError(
            "Server returned aggregate metrics only. "
            "Single-submission evaluations require DEBUG mode."
        )

    metrics = batch_response.get("metrics") or batch_response.get("summary")
    if isinstance(metrics, dict):
        # Attach aggregate context for callers that need it.
        return {**result, "batch_metrics": metrics}
    return result


class EvaluationClient:
    """Blocking client for the evaluation server."""

    def __init__(self, server_url: str, timeout: float = 300.0, api_key: Optional[str] = None) -> None:
        self.server_url = server_url.rstrip("/")
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key
        self.client = httpx.Client(timeout=timeout, headers=headers)

    def check_health(self) -> Dict[str, Any]:
        response = self.client.get(f"{self.server_url}/health")
        response.raise_for_status()
        return response.json()

    def evaluate_batch(
        self,
        submissions: List[Submission],
        batch_metadata: Optional[Dict[str, Any]] = None,
        stream_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        payload = _build_batch_payload(submissions, batch_metadata)
        url = f"{self.server_url}"

        if not stream_progress:
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

        params = {"stream_progress": "true"}
        final_payload: Optional[Dict[str, Any]] = None

        with self.client.stream("POST", url, params=params, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                event_type = event.get("type")

                if event_type == "status":
                    _notify_progress(progress_callback, 0, event.get("total", 0))
                elif event_type == "progress":
                    _notify_progress(progress_callback, event.get("completed", 0), event.get("total", 0))
                elif event_type == "result":
                    payload_data = event.get("payload")
                    if isinstance(payload_data, dict):
                        final_payload = payload_data
                elif event_type == "error":
                    raise RuntimeError(event.get("message", "Unknown streaming error"))
                else:
                    if isinstance(event, dict) and ("accuracy" in event or "timeout_rate" in event):
                        final_payload = event

        if final_payload is None:
            raise RuntimeError("Streaming evaluation finished without a result payload")
        return final_payload

    def evaluate_submission(
        self,
        submission: Submission,
        batch_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        batch_response = self.evaluate_batch([submission], batch_metadata)
        return _extract_single_result(batch_response)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "EvaluationClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class AsyncEvaluationClient:
    """Async counterpart with connection pooling."""

    def __init__(
        self,
        server_url: str,
        timeout: float = 7200.0,
        max_connections: int = 100,
        api_key: Optional[str] = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        limits = Limits(max_connections=max_connections, max_keepalive_connections=20)
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key
        self.client = httpx.AsyncClient(timeout=timeout, limits=limits, headers=headers)

    async def check_health(self) -> Dict[str, Any]:
        response = await self.client.get(f"{self.server_url}/health")
        response.raise_for_status()
        return response.json()

    async def evaluate_batch(
        self,
        submissions: List[Submission],
        batch_metadata: Optional[Dict[str, Any]] = None,
        stream_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        payload = _build_batch_payload(submissions, batch_metadata)
        url = f"{self.server_url}"

        if not stream_progress:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

        params = {"stream_progress": "true"}
        final_payload: Optional[Dict[str, Any]] = None

        async with self.client.stream("POST", url, params=params, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                event_type = event.get("type")

                if event_type == "status":
                    _notify_progress(progress_callback, 0, event.get("total", 0))
                elif event_type == "progress":
                    _notify_progress(progress_callback, event.get("completed", 0), event.get("total", 0))
                elif event_type == "result":
                    payload_data = event.get("payload")
                    if isinstance(payload_data, dict):
                        final_payload = payload_data
                elif event_type == "error":
                    raise RuntimeError(event.get("message", "Unknown streaming error"))
                else:
                    if isinstance(event, dict) and ("accuracy" in event or "timeout_rate" in event):
                        final_payload = event

        if final_payload is None:
            raise RuntimeError("Streaming evaluation finished without a result payload")
        return final_payload

    async def evaluate_submission(
        self,
        submission: Submission,
        batch_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        batch_response = await self.evaluate_batch([submission], batch_metadata)
        return _extract_single_result(batch_response)

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncEvaluationClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


async def evaluate_submission_async(
    server_url: str,
    submission: Submission,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """
    Convenience helper used by the generation script.

    Raises:
        ValueError: If the server is not running in DEBUG mode.
    """
    payload = _build_batch_payload([submission])
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{server_url.rstrip('/')}/evaluate", json=payload)
        response.raise_for_status()
        return _extract_single_result(response.json())
