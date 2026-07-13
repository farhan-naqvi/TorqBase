"""GRAPH 2 — ReAct diagnostic agent (LangGraph).

reason -> [act -> observe -> reason]* -> draft -> verify -> END

Falls back to plain retrieve->answer (Slice 1 path) if the graph misbehaves,
keeping CLAUDE.md rule 6: the demo must always have a working fallback.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langgraph.graph import END, StateGraph

from agent.nodes import (
    AgentState,
    act,
    draft,
    observe,
    reason,
    verify,
    _MAX_LOOPS,
)

log = logging.getLogger(__name__)

_SPEC_RE = re.compile(
    r"\b(rpm|torque|temperature|bar|psi|nm|compression ratio|fuel mixture|timing|clearance)\b|°c",
    re.IGNORECASE,
)
_DIAGNOSTIC_RE = re.compile(
    r"\b(why|cause|won't start|wont start|misfire|overheating|rough idle|fix|troubleshoot)\b",
    re.IGNORECASE,
)

_graph = None  # compiled graph singleton — built once, reused


def _should_act(state: AgentState) -> str:
    """Route from reason: call a tool or go straight to drafting."""
    if state["tool_action"] == "answer" or state["loops"] > _MAX_LOOPS:
        return "draft"
    return "act"


def _should_end(state: AgentState) -> str:
    """Route from verify: accept answer or loop back for another round."""
    if state["grounded"] or state["loops"] > _MAX_LOOPS:
        return END
    return "reason"


def _build_graph():
    g = StateGraph(AgentState)

    g.add_node("reason", reason)
    g.add_node("act", act)
    g.add_node("observe", observe)
    g.add_node("draft", draft)
    g.add_node("verify", verify)

    g.set_entry_point("reason")
    g.add_conditional_edges("reason", _should_act, {"act": "act", "draft": "draft"})
    g.add_edge("act", "observe")
    g.add_edge("observe", "reason")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", _should_end, {END: END, "reason": "reason"})

    return g.compile()


def answer(
    question: str,
    session_id: str = "anon",
    expertise: str = "expert",
    formula_context: list[dict[str, Any]] | None = None,
    mode: str = "general",
) -> dict[str, Any]:
    """Run the ReAct agent and return {answer, citations, confidence, related_questions}.

    Falls back to the plain retrieve->answer path if the graph raises.
    expertise: "beginner" | "expert" — controls answer verbosity and jargon level.
    """
    global _graph
    from config import get_settings
    from ingestion.format_router import detect_language

    settings = get_settings()

    # Load prior conversation turns for context
    history_note = ""
    turns: list[dict[str, Any]] = []
    try:
        from memory.store import get_conversation
        turns = get_conversation(session_id)
        if turns:
            # Summarise the last 4 turns (2 exchanges) as a context note
            recent = turns[-4:]
            history_parts = []
            for t in recent:
                role = t.get("role", "")
                content = str(t.get("content", ""))[:300]
                history_parts.append(f"{role.capitalize()}: {content}")
            history_note = "\n\nPrior conversation:\n" + "\n".join(history_parts)
    except Exception:
        pass

    resolved_question = _resolve_references(question, turns)
    # Append history context to the resolved question so retrieval and drafting understand follow-ups.
    question_with_context = resolved_question + history_note if history_note else resolved_question
    lang = detect_language(question)

    if settings.fast_path_enabled and mode != "formula":
        route = _classify_route(resolved_question)
        if route != "full_agent":
            try:
                result = _fast_answer(route, resolved_question, question_with_context, lang, expertise, formula_context or [])
                return {
                    "answer": result["draft"],
                    "citations": result["citations"],
                    "confidence": "high",
                    "related_questions": [],
                    "conflicts": result.get("conflicts", []),
                }
            except Exception:
                log.exception("graph.answer: fast path failed; falling back to full agent")

    if _graph is None:
        _graph = _build_graph()

    initial: AgentState = {
        "question": question_with_context,
        "lang": lang,
        "expertise": expertise,
        "scratch": [],
        "tool_action": None,
        "tool_args": {},
        "draft": "",
        "citations": [],
        "grounded": False,
        "loops": 0,
        "related": [],
        "formula_context": formula_context or [],
        "mode": mode,
        "scratch_grouped": {},
        "conflicts": [],
    }

    try:
        result = _graph.invoke(initial)
    except Exception:
        log.exception("graph.answer: agent graph failed; falling back to plain retrieval")
        result = _plain_answer(resolved_question)

    return {
        "answer": result["draft"],
        "citations": result["citations"],
        "confidence": "high" if result.get("grounded") else "low",
        "related_questions": result.get("related", []),
        "conflicts": result.get("conflicts", []),
    }


def _resolve_references(question: str, turns: list[dict[str, Any]]) -> str:
    """Expand short follow-up questions into standalone retrieval queries."""
    short_question = len((question or "").split()) < 8
    q_lower = (question or "").lower()
    reference_words = any(w in q_lower for w in [
        "that", "this", "it", "those", "these", "there", "same",
        "what about", "and for", "how about", "the same", "also",
    ])
    if not short_question or not reference_words or not turns:
        return question

    last_user = ""
    last_assistant = ""
    for turn in reversed(turns):
        if turn.get("role") == "assistant" and not last_assistant:
            last_assistant = str(turn.get("content", ""))[:300]
        if turn.get("role") == "user" and not last_user:
            last_user = str(turn.get("content", ""))[:200]
        if last_user and last_assistant:
            break
    if not last_user:
        return question

    try:
        from llm import chat

        expanded = chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a query expander for a technical search system. "
                        "The user asked a follow-up question that contains references "
                        "to the previous exchange. Rewrite ONLY the user's new question "
                        "as a standalone, fully self-contained search query. "
                        "Do not answer the question. Output the rewritten query only. "
                        "Keep it under 25 words."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Previous question: {last_user}\n"
                        f"Previous answer summary: {last_assistant[:200]}\n"
                        f"New question: {question}\n\n"
                        "Rewritten standalone query:"
                    ),
                },
            ],
            temperature=0.0,
        )
        expanded = expanded.strip().strip('"').strip("'")
        if expanded and len(expanded) > 5:
            log.info("graph: expanded %r -> %r", question, expanded)
            return expanded
    except Exception:
        pass
    return question


def _classify_route(question: str) -> str:
    if _SPEC_RE.search(question or ""):
        return "spec_fast"
    if _DIAGNOSTIC_RE.search(question or ""):
        return "kg_fast"
    return "full_agent"


def _diagnostic_symptom(question: str) -> str:
    text = (question or "").lower()
    for symptom in (
        "misfire", "misfiring", "overheating", "rough idle", "loss of power",
        "hard starting", "hard start", "won't start", "wont start", "vibration",
        "won't run", "wont run",
    ):
        if symptom in text:
            return "misfire" if symptom == "misfiring" else symptom
    cleaned = re.sub(r"\b(why|what|is|are|the|engine|motor|cause|causes|fix|troubleshoot|does|do|my|a|an)\b", " ", text)
    return " ".join(cleaned.split()) or question


def _fast_answer(
    route: str,
    question: str,
    question_with_context: str,
    lang: str,
    expertise: str,
    formula_context: list[dict[str, Any]] | None = None,
    mode: str = "general",
) -> dict[str, Any]:
    from agent.nodes import draft
    from agent.tools import diagnostic_tree, hybrid_search, spec_lookup

    scratch: list[dict[str, Any]] = []
    if route == "spec_fast":
        specs = spec_lookup(key=question, engine=None)
        context = hybrid_search(query=question_with_context, k=5)
        scratch.append({"tool": "spec_lookup", "args": {"key": question, "engine": None}, "results": specs})
        scratch.append({"tool": "hybrid_search", "args": {"query": question_with_context, "k": 5}, "results": context})
    elif route == "kg_fast":
        symptom = _diagnostic_symptom(question)
        kg_results = diagnostic_tree(symptom=symptom, engine=None)
        if kg_results:
            scratch.append({"tool": "diagnostic_tree", "args": {"symptom": symptom, "engine": None}, "results": kg_results})
        context = hybrid_search(query=question_with_context, k=8)
        if context:
            try:
                from agent.reranker import rerank

                context = rerank(question, context)
            except Exception:
                pass
        scratch.append({"tool": "hybrid_search", "args": {"query": question_with_context, "k": 8}, "results": context})
    else:
        raise ValueError(f"unknown fast route: {route}")

    state: AgentState = {
        "question": question_with_context,
        "lang": lang,
        "expertise": expertise,
        "scratch": scratch,
        "tool_action": "answer",
        "tool_args": {},
        "draft": "",
        "citations": [],
        "grounded": True,
        "loops": 0,
        "related": [],
        "formula_context": formula_context or [],
        "mode": mode,
    }
    return draft(state)


def _plain_answer(question: str) -> dict[str, Any]:
    """Plain retrieve -> answer fallback (CLAUDE.md rule 6)."""
    from agent.retriever_hybrid import search
    from agent.nodes import _clean_answer_for_user, _clean_chunk_for_llm
    from config import get_settings
    from llm import chat

    settings = get_settings()
    chunks = search(question, k=settings.rerank_top_k)

    if not chunks:
        return {
            "draft": "I don't have enough information in the knowledge base to answer that question.",
            "citations": [],
            "grounded": True,
            "related": [],
        }

    context_lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for i, c in enumerate(chunks, 1):
        source_refs = c.get("source_refs") or [{}]
        ref = source_refs[0] if source_refs else {}
        metadata = c.get("metadata") or {}
        label = ref.get("filename") or c.get("doc_id") or "unknown"
        page = ref.get("page", "")
        cite_label = f"{label} p.{page}" if page else label
        source_type = metadata.get("source_type", "")
        cleaned_content = _clean_chunk_for_llm(c["content"])
        if source_type == "expert_correction":
            context_lines.append(f"[{i}] EXPERT CORRECTION - human verified\n{cleaned_content}")
            label = "Expert Correction"
        else:
            context_lines.append(f"[{i}] From {label} (page {page or '?'})\n{cleaned_content}")
        citations.append({
            "n": i,
            "id": c.get("id"),
            "chunk_id": c.get("id"),
            "doc_id": c.get("doc_id", ""),
            "filename": label,
            "page": page,
            "snippet": cleaned_content[:200],
            "source_type": source_type or None,
        })

    answer_text = chat(
        [
            {
                "role": "system",
                "content": (
                    "You are TwoStrokeGPT, an expert on two-stroke engines. "
                    "Synthesize a clear answer from the numbered sources below. "
                    "Never paste raw source text or internal markers. Cite sources at the end only. Never invent numbers. "
                    "Sources marked '✓ EXPERT CORRECTION' are human-verified and take precedence if sources conflict. "
                    "Write a COMPLETE answer. If you cannot fit everything, "
                    "summarise remaining points in a short final paragraph — "
                    "never end mid-sentence.\n\n"
                    + "\n\n".join(context_lines)
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.1,
        max_tokens=2000,
    )

    return {"draft": _clean_answer_for_user(answer_text), "citations": citations, "grounded": False, "related": []}
