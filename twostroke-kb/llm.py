"""Unified LLM client. All model calls go through here.

Config is driven by config.yaml + .env — change LLM_MODEL in .env
and every call site picks it up without any code changes.

  from llm import chat, stream_chat, describe_image
  text = chat([{"role": "user", "content": "Hallo"}])
  data = chat_json([...])
  for tok in stream_chat([...]): ...
  caption = describe_image(png_bytes)

no_think=true in config.yaml strips <think>…</think> blocks from Qwen3
reasoning output so they never appear in answers.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator

from config import get_llm_config

Message = dict[str, str]

# Regex to strip Qwen3 chain-of-thought blocks before returning text
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _maybe_no_think(messages: list[Message]) -> list[Message]:
    """Prepend /no_think to the last user message when no_think=true.

    Ollama's Qwen3 respects this flag to disable chain-of-thought output,
    saving tokens and keeping answers clean.
    """
    cfg = get_llm_config()
    if not cfg.get("no_think"):
        return messages
    msgs = list(messages)
    # Find last user message and prepend the flag
    for i in reversed(range(len(msgs))):
        if msgs[i].get("role") == "user":
            content = msgs[i]["content"]
            if not content.startswith("/no_think"):
                msgs[i] = {**msgs[i], "content": "/no_think\n" + content}
            break
    return msgs


def chat(
    messages: list[Message],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> str:
    """Send a chat completion and return the assistant text."""
    cfg = get_llm_config()
    provider = cfg["provider"].lower()
    _model = model or cfg["model"]
    _temp  = temperature if temperature is not None else cfg["temperature"]
    _maxt  = max_tokens  if max_tokens  is not None else cfg["max_tokens"]

    msgs = _maybe_no_think(messages)

    if provider in ("openai", "local", "ollama", "vllm"):
        try:
            raw = _openai_compat(msgs, _temp, _maxt, _model, cfg)
        except Exception:
            if provider in ("local", "ollama", "vllm"):
                raw = _offline_extractive(msgs)
            else:
                raise
    elif provider == "anthropic":
        raw = _anthropic(msgs, _temp, _maxt, _model, cfg)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r} — set LLM_PROVIDER in .env")

    return _strip_think(raw) if cfg.get("no_think") else raw


def chat_json(messages: list[Message], **kwargs: Any) -> Any:
    """Like chat() but parse the reply as JSON."""
    raw = chat(messages, **kwargs)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = min((raw.find(c) for c in "[{" if raw.find(c) != -1), default=-1)
        end   = max(raw.rfind("]"), raw.rfind("}"))
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start: end + 1])
        raise


# --- providers ---------------------------------------------------------------

def _openai_compat(messages, temperature, max_tokens, model, cfg) -> str:
    """OpenAI SDK pointed at cfg['base_url'] — works for Ollama /v1, vLLM, LM Studio, etc."""
    from openai import OpenAI

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content or ""


def _anthropic(messages, temperature, max_tokens, model, cfg) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.get("api_key") or "")
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo  = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model, system=system or None,
        messages=convo, temperature=temperature, max_tokens=max_tokens,
    )
    return "".join(block.text for block in resp.content if block.type == "text")


# --- streaming ---------------------------------------------------------------

def stream_chat(
    messages: list[Message],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> Iterator[str]:
    """Yield text tokens as they arrive. Strips <think> blocks on the fly."""
    cfg      = get_llm_config()
    provider = cfg["provider"].lower()
    _model   = model or cfg["model"]
    _temp    = temperature if temperature is not None else cfg["temperature"]
    _maxt    = max_tokens  if max_tokens  is not None else cfg["max_tokens"]
    _no_think = cfg.get("no_think", False)

    msgs = _maybe_no_think(messages)

    if provider in ("openai", "local", "ollama", "vllm"):
        try:
            yield from _openai_compat_stream(msgs, _temp, _maxt, _model, cfg, _no_think)
        except Exception:
            if provider not in ("local", "ollama", "vllm"):
                raise
            for token in _offline_extractive(msgs).split(" "):
                yield token + " "
    elif provider == "anthropic":
        yield from _anthropic_stream(msgs, _temp, _maxt, _model, cfg, _no_think)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")


def _openai_compat_stream(messages, temperature, max_tokens, model, cfg, no_think) -> Iterator[str]:
    from openai import OpenAI

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    stream = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens, stream=True,
    )
    in_think = False
    buf = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        if no_think:
            buf += delta
            # Stream-strip <think>...</think> as tokens arrive
            while True:
                if not in_think:
                    start = buf.find("<think>")
                    if start == -1:
                        # No open tag — yield everything up to a partial match guard
                        safe = buf if "<" not in buf else buf[:buf.rfind("<")]
                        if safe:
                            yield safe
                            buf = buf[len(safe):]
                        break
                    else:
                        if start > 0:
                            yield buf[:start]
                        buf = buf[start + len("<think>"):]
                        in_think = True
                else:
                    end = buf.find("</think>")
                    if end == -1:
                        buf = ""  # discard buffered think content
                        break
                    else:
                        buf = buf[end + len("</think>"):]
                        in_think = False
        else:
            yield delta
    if buf and not in_think:
        yield buf


def _anthropic_stream(messages, temperature, max_tokens, model, cfg, no_think) -> Iterator[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.get("api_key") or "")
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    convo = [m for m in messages if m["role"] != "system"]

    with client.messages.stream(
        model=model, system="\n".join(system_parts) if system_parts else "",
        messages=convo, temperature=temperature, max_tokens=max_tokens,
    ) as s:
        for tok in s.text_stream:
            yield tok


def _offline_extractive(messages: list[Message]) -> str:
    """Conservative local fallback for demos when Ollama is not running."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")

    if "Available tools" in system:
        return json.dumps({
            "thought": "Search uploaded documents for grounded evidence.",
            "action": "hybrid_search",
            "args": {"query": user.replace("Question:", "").strip()},
        })

    sources = _extract_numbered_sources(system)
    if not sources:
        return "I don't have enough information in the uploaded documents to answer that question."

    terms = _keywords(user)
    ranked = sorted(sources, key=lambda s: _overlap_score(terms, s["text"]), reverse=True)
    selected = [s for s in ranked if _overlap_score(terms, s["text"]) > 0][:3] or ranked[:2]
    lines = []
    for source in selected:
        sentence = _best_sentence(source["text"], terms)
        if sentence:
            lines.append(f"{sentence} [Source {source['n']}]")

    if not lines:
        return "I found relevant uploaded sources, but not enough clear text to answer confidently."
    return "From the uploaded documents, the relevant point is:\n\n" + "\n".join(
        f"- {line}" for line in lines
    )


def _extract_numbered_sources(text: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"\[Source\s+(\d+)\]\s*(?:\([^)]+\))?\s*\n(.*?)(?=\n\s*\[Source\s+\d+\]|\Z)",
        re.S,
    )
    return [{"n": m.group(1), "text": _clean_space(m.group(2))} for m in pattern.finditer(text)]


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[A-Za-zÄÖÜäöüß0-9][A-Za-zÄÖÜäöüß0-9_.-]{2,}", text.lower())
    stop = {
        "the", "and", "for", "what", "does", "about", "with", "from", "that",
        "this", "how", "are", "ist", "und", "der", "die", "das", "was",
        "sample", "document", "documents", "uploaded",
    }
    return {word for word in words if word not in stop}


def _overlap_score(query_terms: set[str], text: str) -> int:
    text_l = text.lower()
    return sum(1 for term in query_terms if term in text_l)


def _best_sentence(text: str, query_terms: set[str]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    candidates = [_clean_space(sentence) for sentence in sentences if _clean_space(sentence)]
    if not candidates:
        return _clean_space(text[:280])
    best = max(candidates, key=lambda s: (_overlap_score(query_terms, s), min(len(s), 280)))
    if len(best) > 360:
        best = best[:357].rstrip() + "..."
    return best


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --- vision ------------------------------------------------------------------

def describe_image(
    image_bytes: bytes,
    prompt: str = (
        "This is a technical diagram or figure from a two-stroke engine manual. "
        "Describe precisely what you see: component names, labels, measurements, "
        "arrows, callouts, and any numeric values. Focus on technical content only."
    ),
) -> str:
    """Send image bytes to the vision LLM and return a text description.

    Vision model is LLM_VISION_MODEL from .env, falling back to LLM_MODEL.
    Supports local (Ollama /v1), vLLM, OpenAI, and Anthropic.
    """
    import base64

    b64  = base64.b64encode(image_bytes).decode()
    cfg  = get_llm_config()
    provider = cfg["provider"].lower()
    model    = cfg["vision_model"]   # falls back to main model if not set separately

    if provider in ("openai", "local", "ollama", "vllm"):
        from openai import OpenAI
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""

    if provider == "anthropic":
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=cfg.get("api_key") or "")
        resp = client.messages.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=512,
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    raise ValueError(
        f"describe_image: provider {provider!r} does not support vision. "
        "Set LLM_PROVIDER to openai, local, ollama, vllm, or anthropic."
    )
