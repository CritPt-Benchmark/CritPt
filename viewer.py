"""
Streamlit viewer for CritPt generation results.

Run:
    streamlit run viewer.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

RESULTS_DIR = Path(__file__).parent / "results" / "generations"

ROLE_ICONS = {"system": "⚙️", "user": "🧑", "assistant": "🤖", "tool": "🛠️"}


def extract_text(msg: dict[str, Any]) -> str:
    """Pull text out of a chat message — handles both `content` (str) and `contents` (list of parts)."""
    if isinstance(msg.get("content"), str):
        return msg["content"]
    parts = msg.get("contents") or []
    chunks = []
    for p in parts:
        if isinstance(p, dict) and p.get("content_text"):
            chunks.append(p["content_text"])
        elif isinstance(p, str):
            chunks.append(p)
    return "\n\n".join(chunks)


def find_runs(base: Path) -> list[Path]:
    """A 'run' is any directory that contains a main.json artifact."""
    if not base.exists():
        return []
    return sorted({p.parent for p in base.rglob("main.json")})


def parse_run_path(run_dir: Path) -> dict[str, str]:
    """Decode the deeply-nested folder convention into labeled fields."""
    rel = run_dir.relative_to(RESULTS_DIR).parts
    out: dict[str, str] = {"problem": rel[0] if rel else "?", "epoch": rel[-1] if rel else "?"}
    if len(rel) >= 2:
        out["model"] = rel[-2]
    for part in rel[1:-2]:
        if "_" in part:
            key, _, val = part.rpartition("_")
            out[key] = val
    return out


def load_submission(run_dir: Path, problem_id: str) -> dict | None:
    f = run_dir / f"{problem_id}_main.json"
    if f.exists():
        return json.loads(f.read_text())
    # Fall back: any *_main.json in the dir
    for cand in run_dir.glob("*_main.json"):
        return json.loads(cand.read_text())
    return None


# ---------- UI ----------

st.set_page_config(page_title="CritPt Results", layout="wide")
st.title("CritPt — Generation Viewer")

runs = find_runs(RESULTS_DIR)
if not runs:
    st.warning(f"No runs found under {RESULTS_DIR}. Run `python -m critpt generate ...` first.")
    st.stop()

# Build label → path mapping for the sidebar
labeled = []
for r in runs:
    meta = parse_run_path(r)
    label = f"{meta['problem']}  ·  {meta.get('model', '?')}  ·  epoch {meta['epoch']}"
    labeled.append((label, r, meta))

with st.sidebar:
    st.header("Runs")
    idx = st.selectbox(
        "Select a run",
        range(len(labeled)),
        format_func=lambda i: labeled[i][0],
    )

label, run_dir, meta = labeled[idx]

# ---------- Header / metadata ----------

st.subheader(meta["problem"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", meta.get("model", "?"))
c2.metric("Epoch", meta["epoch"])
c3.metric("Parsing", meta.get("parsing", "?"))
c4.metric("Multiturn w/ ans", meta.get("multiturn_with_answer", "?"))

# ---------- Load artifacts ----------

main_path = run_dir / "main.json"
artifact = json.loads(main_path.read_text())
submission = load_submission(run_dir, meta["problem"])

tab_chat, tab_answer, tab_meta, tab_raw = st.tabs(
    ["Conversation", "Final answer", "Run metadata", "Raw JSON"]
)

with tab_chat:
    chat = artifact.get("chat") or []
    if not chat:
        st.info("No chat messages on this run.")
    for msg in chat:
        role = msg.get("role", "unknown")
        icon = ROLE_ICONS.get(role, "•")
        with st.chat_message(role if role in {"user", "assistant", "system"} else "assistant"):
            st.caption(f"{icon} {role}")
            text = extract_text(msg)
            if text:
                st.markdown(text)
            else:
                st.code(json.dumps(msg, indent=2), language="json")

with tab_answer:
    if submission is None:
        st.info("No submission file found in this run directory.")
    else:
        st.caption(f"problem_id: `{submission.get('problem_id')}`")
        st.caption(f"timestamp: {submission.get('timestamp')}")
        code = submission.get("generated_code") or ""
        st.markdown("### Generated answer")
        if code:
            st.markdown(code)
            with st.expander("Show as plain text"):
                st.code(code, language="markdown")
        else:
            st.info("Empty `generated_code` field.")

with tab_meta:
    st.markdown("**Run directory**")
    st.code(str(run_dir))
    st.markdown("**Parsed config**")
    st.json(meta)
    st.markdown("**Artifact header**")
    header = {k: artifact.get(k) for k in ("artifact_type", "version", "step", "created_at")}
    st.json(header)
    if submission:
        st.markdown("**Submission generation_config**")
        st.json(submission.get("generation_config") or {})

with tab_raw:
    pick = st.radio("File", ["main.json", "submission"], horizontal=True)
    if pick == "main.json":
        st.json(artifact)
    else:
        st.json(submission or {})
