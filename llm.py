"""
One place to call Claude for a structured answer.

Asks Claude to call a named tool. Some models don't accept a *forced* tool choice, so this
first tries tool_choice={"type": "tool"}, and if the API rejects that, retries with
tool_choice auto plus an explicit instruction — and as a last resort parses JSON from text.
"""

from __future__ import annotations

import json
import re
import sys

_forced_ok: dict[str, bool] = {}


def tool_call(model: str, tool: dict, messages: list[dict], system: str | None = None,
              max_tokens: int = 1200) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    name = tool["name"]
    kw = {"model": model, "max_tokens": max_tokens, "tools": [tool], "messages": messages}
    if system:
        kw["system"] = system

    if _forced_ok.get(model, True):
        try:
            msg = client.messages.create(tool_choice={"type": "tool", "name": name}, **kw)
            out = _extract(msg, name)
            if out is not None:
                return out
        except anthropic.BadRequestError as e:
            if "tool_choice" not in str(e):
                raise
            _forced_ok[model] = False   # this model doesn't support forced tools; don't try again

    nudge = (f"\n\nYou MUST respond by calling the `{name}` tool with your answer. "
             "Do not reply with plain text.")
    msgs = [dict(m) for m in messages]
    if msgs and isinstance(msgs[-1].get("content"), str):
        msgs[-1]["content"] += nudge
    msg = client.messages.create(**{**kw, "messages": msgs, "tool_choice": {"type": "auto"}})
    out = _extract(msg, name)
    if out is not None:
        return out
    # last resort: a JSON object in the text
    text = "".join(getattr(b, "text", "") for b in msg.content)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    print(f"llm: no tool call in reply: {text[:200]}", file=sys.stderr)
    raise RuntimeError("Claude returned no structured answer")


def _extract(msg, name: str) -> dict | None:
    for b in msg.content:
        if getattr(b, "type", "") == "tool_use" and b.name == name:
            return dict(b.input)
    return None
