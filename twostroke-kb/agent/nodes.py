"""LangGraph nodes for the ReAct agent. Each takes the shared state dict and
returns a dict of keys to update (LangGraph merges them into state).

State lifecycle:
  reason -> [act -> observe -> reason]* -> draft -> verify -> END
"""
from __future__ import annotations

import re
import logging
from typing import Any

try:
    from typing import NotRequired, TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore[assignment]
    from typing_extensions import NotRequired


class AgentState(TypedDict):
    question: str
    lang: str
    expertise: str                  # "beginner" | "expert" â€” controls answer verbosity
    scratch: list[dict[str, Any]]   # accumulated tool call results
    tool_action: str | None         # "hybrid_search" | "spec_lookup" | "answer"
    tool_args: dict[str, Any]       # args for the next tool call
    draft: str                      # composed answer text
    citations: list[dict[str, Any]] # citation objects for the API response
    grounded: bool                  # verifier result
    loops: int                      # iteration counter (guard against infinite loops)
    related: list[str]              # suggested follow-up questions
    formula_context: NotRequired[list[dict[str, Any]]]
    mode: NotRequired[str]
    gap_warning: NotRequired[str]
    scratch_grouped: NotRequired[dict[str, list[dict[str, Any]]]]
    conflicts: NotRequired[list[dict[str, Any]]]


_MAX_LOOPS = 3
log = logging.getLogger(__name__)
_VERIFY_NUMERIC_RE = re.compile(
    r"\d+\.?\d*\s*(rpm|nm|bar|Â°c|psi|mm|kg)\b",
    re.IGNORECASE,
)

_TOOL_SCHEMA = """\
Available tools (respond with ONLY valid JSON, no prose):

  hybrid_search(query: str)
    - semantic + keyword search over all uploaded documents
    - use for: general questions, definitions, procedures, specifications

  spec_lookup(key: str, engine: str|null)
    - exact numeric values from spreadsheets (RPM, torque, temperature, etc.)
    - use for: any question asking for a specific number or measurement

  diagnostic_tree(symptom: str, engine: str|null)
    - walks the Knowledge Graph: symptom -> cause -> fix
    - use for: troubleshooting questions (why, won't start, misfire, overheating,
      rough idle, loss of power, hard starting, vibration)
    - engine param: engine model name if mentioned, else null

  graph_lookup(entity: str, relation: str|null)
    - finds related parts, symptoms, causes in the Knowledge Graph
    - use for: "what is related to X", "what causes Y", "what parts affect Z"

When you have enough information to answer:
  {"thought": "...", "action": "answer", "args": {}}

Tool call format:
  {"thought": "...", "action": "hybrid_search",  "args": {"query": "..."}}
  {"thought": "...", "action": "spec_lookup",     "args": {"key": "...", "engine": null}}
  {"thought": "...", "action": "diagnostic_tree", "args": {"symptom": "...", "engine": null}}
  {"thought": "...", "action": "graph_lookup",    "args": {"entity": "...", "relation": null}}
"""

_DIAGNOSTIC_HINT_RE = re.compile(
    r"\b(why|cause|won't start|wont start|misfire|overheating|rough idle|loss of power|hard start|vibration|won't run|wont run|fails|failure)\b",
    re.IGNORECASE,
)
_DIAGNOSTIC_ANSWER_RE = re.compile(
    r"\b(why|cause|won't start|wont start|misfire|overheating|rough idle|loss of power|hard start|vibration|won't run|wont run|fails|failure|problem|issue|fault)\b",
    re.IGNORECASE,
)


GENERAL_SYSTEM_PROMPT = """
You are TorqBase, an expert engineering assistant for two-stroke aircraft engines.
You have retrieved relevant excerpts from technical documents.
Your job is to synthesize those excerpts into a clear, professional answer -
NOT to copy-paste the excerpts.

CRITICAL RULES:
1. Write a real answer in your own words. Never paste raw document text.
2. Remove ALL internal markers from your answer:
   - Never include "--- Slide N ---", "--- Page N ---", "[Source N]" in the answer body.
   - These are internal references only. Use clean inline citations like 1, 2, 3 only in the Sources section.
3. Structure your answer clearly. Use markdown formatting.
4. Be direct. Start with the answer, then explain.
5. If sources conflict on a numeric value, say so explicitly.
6. If you don't have enough information, say so. Never guess.
7. Keep answers focused. Do not pad with filler text.
8. Cite sources at the END only, not scattered through the answer body.
9. Sources marked 'EXPERT CORRECTION' are human-verified and take precedence over document sources if they conflict.
10. IMPORTANT: If the sources contain ANY information related to the question,
you MUST use it to write an answer. Only say you cannot find information
if the sources contain absolutely nothing relevant - not even indirectly.
Regulatory text, standards, and technical specifications ARE valid sources.
A source about '§ 33.49 Endurance test' IS relevant to 'what is an endurance test'.

ANSWER FORMAT:

### [Short title that directly answers the question]

[Direct answer in 1-2 sentences. State the key fact immediately.]

[Supporting explanation in clear paragraphs. Use bullet points for lists
of components, steps, or specifications. Use a table for comparing values
across sources or listing multiple specs.]

**Key specifications** (only if numeric specs appear in sources):
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
[one row per spec - omit this table if no specs are present]

---
**Sources**
[1] Filename - Page/Slide N - one sentence on what it contributed
[2] ...
[omit sources that were not actually used in the answer]
"""


DIAGNOSTIC_SYSTEM_PROMPT = """
You are TorqBase, an expert engineering assistant for two-stroke aircraft engines.

CRITICAL RULES:
1. Write a real answer in your own words. Never paste raw document text.
2. Never include "--- Slide N ---", "--- Page N ---", "[Source N]" in the answer body.
3. Structure diagnostic answers as: Cause -> Explanation -> Action.
4. Be direct and actionable.
5. Cite sources at the END only, not scattered through the answer body.
6. IMPORTANT: If the sources contain ANY information related to the question,
you MUST use it to write an answer. Only say you cannot find information
if the sources contain absolutely nothing relevant - not even indirectly.
Regulatory text, standards, and technical specifications ARE valid sources.
A source about '§ 33.49 Endurance test' IS relevant to 'what is an endurance test'.

ANSWER FORMAT:

### [Symptom / Problem Title]

**Most likely cause:** [direct statement]

**Explanation:**
[Clear explanation of why this happens, in your own words, 2-4 sentences]

**Recommended actions:**
1. [First action]
2. [Second action]
3. [...]

**Additional causes to consider:**
- [Other possible cause and brief explanation]
- [...]

---
**Sources**
[1] Filename - Page N
[2] ...
"""


def _clean_chunk_for_llm(content: str) -> str:
    """Remove internal document markers that should never appear in answers."""
    content = str(content or "")
    content = re.sub(r"---\s*Slide\s*\d+\s*---", "", content, flags=re.IGNORECASE)
    content = re.sub(r"---\s*Page\s*\d+\s*---", "", content, flags=re.IGNORECASE)
    content = re.sub(r"GE PROPRIETARY INFORMATION[^\n]*\n?", "", content, flags=re.IGNORECASE)
    content = re.sub(r"@#GEINT&\*[^\n]*\n?", "", content, flags=re.IGNORECASE)
    content = re.sub(r"GE Internal[^\n]*\n?", "", content, flags=re.IGNORECASE)
    content = re.sub(r"For internal distribution only[^\n]*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[ \t]{2,}", " ", content)
    return content.strip()


def _clean_answer_for_user(content: str) -> str:
    content = _clean_chunk_for_llm(content)
    content = re.sub(r"\[Source\s+\d+\]", "", content, flags=re.IGNORECASE)
    return content.strip()

def _scratch_summary(scratch: list[dict[str, Any]]) -> str:
    if not scratch:
        return ""
    parts: list[str] = []
    for i, item in enumerate(scratch, 1):
        tool = item.get("tool", "result")
        results = item.get("results", [])
        excerpts = []
        for r in results[:3]:
            if isinstance(r, dict):
                excerpts.append(r.get("content") or r.get("value") or str(r))
            else:
                excerpts.append(str(r))
        parts.append(f"[Tool {i}: {tool}]\n" + "\n".join(excerpts))
    return "\n\n".join(parts)


def reason(state: AgentState) -> dict[str, Any]:
    """LLM decides: enough info to answer, or which tool to call next."""
    from llm import chat_json

    summary = _scratch_summary(state["scratch"])
    gap_warning = ""
    if state.get("loops", 0) == 0:
        similar_gaps = _check_known_gaps(state["question"])
        if similar_gaps:
            gap_warning = (
                "Note: Similar questions previously had weak evidence in the knowledge base: "
                + "; ".join(f'"{g}"' for g in similar_gaps)
                + ". Search carefully and flag if evidence is thin."
            )
    system = (
        "You are a ReAct reasoning agent for two-stroke engine knowledge.\n"
        + _TOOL_SCHEMA
        + ("\n\nTool results so far:\n" + summary if summary else "")
    )
    if gap_warning:
        system += "\n\n" + gap_warning
    if state.get("loops", 0) == 0 and _DIAGNOSTIC_HINT_RE.search(state["question"]):
        system += (
            "\n\nNote: This looks like a diagnostic question. Consider using diagnostic_tree "
            "first to find symptom->cause->fix paths from the Knowledge Graph, then "
            "hybrid_search for supporting documentation."
        )

    decision = chat_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {state['question']}"},
        ],
        max_tokens=256,
    )

    action = str(decision.get("action", "answer"))
    args = decision.get("args") or {}
    if not isinstance(args, dict):
        args = {}

    # Guard: force answer after max loops
    if state["loops"] >= _MAX_LOOPS:
        action = "answer"

    return {
        "tool_action": action,
        "tool_args": args,
        "loops": state["loops"] + 1,
        **({"gap_warning": gap_warning} if gap_warning else {}),
    }


def _check_known_gaps(question: str) -> list[str]:
    """Return unresolved gap questions with simple keyword overlap."""
    try:
        from config import get_connection

        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT question FROM gaps WHERE resolved = false ORDER BY created_at DESC LIMIT 100")
            gap_questions = [str(r[0] or "") for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception:
        return []

    stop = {"what", "is", "the", "a", "an", "of", "for", "in", "how", "does", "do", "are"}
    question_words = set(question.lower().split()) - stop
    similar: list[str] = []
    for gap_question in gap_questions:
        gap_words = set(gap_question.lower().split()) - stop
        if len(question_words & gap_words) >= 3:
            similar.append(gap_question)
    return similar[:3]


def act(state: AgentState) -> dict[str, Any]:
    """Dispatch the chosen tool and append its result to scratch."""
    from agent.tools import TOOLS

    tool_name = state["tool_action"]
    args = state.get("tool_args") or {}

    if tool_name == "answer" or tool_name not in TOOLS:
        return {}

    try:
        results = TOOLS[tool_name](**args)
        if not isinstance(results, list):
            results = [results]
        if tool_name in ("diagnostic_tree", "graph_lookup") and results:
            for r in results:
                if isinstance(r, dict) and "content" not in r:
                    r["content"] = str(r)

        # Transparently rerank after hybrid_search so downstream nodes always
        # see the highest-quality results first.
        if tool_name == "hybrid_search" and results:
            try:
                from agent.reranker import rerank
                query = args.get("query") or state["question"]
                results = rerank(query, results)
            except Exception:
                pass  # reranker unavailable â€” dense-only ranking is fine
    except NotImplementedError:
        results = [{"content": f"Tool '{tool_name}' is not yet implemented.", "error": True}]
    except Exception as exc:
        results = [{"content": f"Tool error: {exc}", "error": True}]

    update: dict[str, Any] = {"scratch": state["scratch"] + [{"tool": tool_name, "args": args, "results": results}]}
    if tool_name in ("hybrid_search", "spec_lookup"):
        try:
            from agent.conflict_detector import group_by_doc

            grouped = dict(state.get("scratch_grouped", {}))
            for doc_id, doc_results in group_by_doc(results).items():
                grouped.setdefault(doc_id, []).extend(doc_results)
            update["scratch_grouped"] = grouped
        except Exception:
            pass
    return update


def observe(state: AgentState) -> dict[str, Any]:
    """Hook for post-tool observation (e.g. summarising a long result set).

    Currently a pass-through: reason() already reads scratch directly.
    """
    return {}


def _format_conflict_section(conflicts: list[dict[str, Any]]) -> str:
    if not conflicts:
        return ""
    lines = ["\n\nWARNING: CONFLICTING VALUES DETECTED ACROSS SOURCES:"]
    for i, conflict in enumerate(conflicts, 1):
        lines.append(f"\nConflict {i} - unit: {conflict.get('unit', '')}")
        for value in conflict.get("values", []):
            lines.append(
                f"  - {value.get('value')} {value.get('unit', '')} in [{value.get('filename')}] "
                f"page {value.get('page') or '?'} - \"{str(value.get('snippet', ''))[:80]}...\""
            )
    lines.append(
        "\nYou MUST explicitly flag these conflicts in your answer. Do not pick one value and ignore the others. "
        "Tell the engineer which document says what and recommend verification against the primary specification document."
    )
    return "\n".join(lines)

    answer_text = chat(
        [
            {
                "role": "system",
                "content": (
                    "You are TwoStrokeGPT, an expert on two-stroke engines. "
                    "Answer ONLY using the numbered sources provided below. "
                    "Cite each fact as [Source N] immediately after the claim. "
                    "If a numeric value (RPM, temperature, timing, torque, etc.) "
                    "is NOT explicitly stated in a source, say you don't know â€” "
                    "never invent or estimate a value. "
                    "Write a COMPLETE answer. If you cannot fit everything, "
                    "summarise the remaining points in a short final paragraph â€” "
                    "never end mid-sentence."
                    + " When [Formula N] sources are provided, present the formula clearly, define variables when stated, state units only when explicit, and cite the source document/page."
                    + style_note
                    + lang_note
                    + f"\n\nSources:\n{context}"
                ),
            },
            {"role": "user", "content": state["question"]},
        ],
        temperature=0.1,
    )

    return {"draft": answer_text, "citations": citations}


def draft(state: AgentState) -> dict[str, Any]:
    """Compose a cited answer in the user's language from all accumulated evidence."""
    import re
    from llm import chat

    evidence: list[dict[str, Any]] = []
    for item in state["scratch"]:
        for r in item.get("results", []):
            if isinstance(r, dict) and not r.get("error"):
                enriched = dict(r)
                enriched.setdefault("tool", item.get("tool"))
                evidence.append(enriched)

    for f in state.get("formula_context", []):
        formula = f.get("formula_latex") or f.get("formula_text") or ""
        if not formula:
            continue
        evidence.append({
            "content": "\n".join(
                part for part in [
                    formula,
                    f.get("context_before") or "",
                    f.get("context_after") or "",
                ]
                if part
            ),
            "doc_id": f.get("doc_id", ""),
            "source_refs": [{
                "filename": f.get("source_title") or f.get("doc_id") or "unknown",
                "page": f.get("page_or_slide") or "",
            }],
            "id": f.get("chunk_id"),
            "chunk_id": f.get("chunk_id"),
        })

    if not evidence:
        return {
            "draft": "I don't have enough information in the knowledge base to answer that question.",
            "citations": [],
        }

    def _clean_chunk(text: str) -> str:
        text = str(text or "")
        text = re.sub(r"-{2,}\s*Slide\s*\d+\s*-{2,}", "", text, flags=re.IGNORECASE)
        text = re.sub(r"-{2,}\s*Page\s*\d+\s*-{2,}", "", text, flags=re.IGNORECASE)
        text = re.sub(r"GE PROPRIETARY INFORMATION[^\n]*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"@#GEINT&\*[^\n]*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"GE Internal\s*[-\u2013]?[^\n]*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"For internal distribution only[^\n]*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[Source\s+\d+\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    context_lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for c in evidence:
        n = len(citations) + 1
        if c.get("tool") in ("diagnostic_tree", "graph_lookup"):
            cleaned = _clean_chunk(c.get("content", ""))
            context_lines.append(f"[{n}] Knowledge Graph:\n{cleaned}")
            citations.append({
                "n": n,
                "doc_id": c.get("doc_id", "knowledge-graph"),
                "filename": "Knowledge Graph",
                "page": "",
                "snippet": cleaned[:200],
                "source_type": "kg",
            })
            continue

        source_refs = c.get("source_refs") or [{}]
        ref = source_refs[0] if source_refs else {}
        label = ref.get("filename") or c.get("doc_id") or "unknown"
        page = ref.get("page", "")
        metadata = c.get("metadata") or {}
        source_type = metadata.get("source_type") or c.get("source_type") or ""
        cleaned = _clean_chunk(c.get("content", c.get("value", "")))
        if source_type == "expert_correction":
            label = "Expert Correction"
            context_lines.append(f"[{n}] EXPERT CORRECTION - human verified:\n{cleaned}")
        else:
            context_lines.append(f"[{n}] {label}{f' p.{page}' if page else ''}:\n{cleaned}")
        citations.append({
            "n": n,
            "id": c.get("id") or c.get("chunk_id"),
            "chunk_id": c.get("id") or c.get("chunk_id"),
            "doc_id": c.get("doc_id", ""),
            "filename": label,
            "page": page,
            "snippet": cleaned[:200],
            "source_type": source_type or None,
        })

    lang = state.get("lang", "en")
    lang_note = f"\n\nAnswer in {lang}." if lang not in ("en", "unknown", "") else ""
    context = "\n\n---\n\n".join(context_lines)
    expertise = state.get("expertise", "expert")
    is_beginner = expertise == "beginner"

    question_lower = state["question"].lower()
    is_diagnostic = any(w in question_lower for w in [
        "why", "cause", "won't start", "wont start", "misfire", "overheating",
        "rough idle", "loss of power", "hard start", "vibration", "won't run",
        "fails", "failure", "problem", "issue", "fault", "symptom",
    ])
    is_formula = state.get("mode", "general") == "formula"

    if is_formula:
        system_prompt = (
            "You are TorqBase, a precise engineering assistant for two-stroke engines.\n\n"
            "The user wants a formula. Using ONLY the sources provided, give a clean structured answer.\n\n"
            "FORMAT YOUR ANSWER EXACTLY LIKE THIS:\n\n"
            "### [Formula Name]\n\n"
            "**Formula:**\n"
            "[Write the formula in plain ASCII math, e.g. BTE = P_b / (m_F x LHV)]\n\n"
            "**Variables:**\n"
            "- Symbol: Full name - unit\n"
            "(list every variable in the formula)\n\n"
            "**What it means:**\n"
            "[One short paragraph in plain English]\n\n"
            "**Example:** (only if an example appears in the sources)\n"
            "[Show the example values from the source]\n\n"
            "---\n"
            "**Sources**\n"
            "[1] Filename - brief note on what it contributed\n\n"
            "RULES:\n"
            "- Never invent a value.\n"
            "- Write the formula in plain ASCII math - no Unicode math symbols.\n"
            "- Do not paste raw source text into the answer.\n"
            "- IMPORTANT: If the sources contain ANY information related to the question, use it. "
            "Only say you cannot find information if the sources contain absolutely nothing relevant."
            + lang_note
        )
    elif is_diagnostic:
        system_prompt = (
            "You are TorqBase, a precise engineering assistant for two-stroke engines.\n\n"
            "The user has a diagnostic or troubleshooting question.\n"
            "Using ONLY the sources provided, give a clear actionable answer.\n\n"
            "FORMAT YOUR ANSWER EXACTLY LIKE THIS:\n\n"
            "### [Problem or Symptom Title]\n\n"
            "**Most likely cause:** [direct one-sentence statement]\n\n"
            "**Explanation:**\n"
            "[2-4 sentences explaining why this happens, in your own words]\n\n"
            "**Recommended actions:**\n"
            "1. [First action]\n"
            "2. [Second action]\n"
            "3. [Continue as needed]\n\n"
            "**Other causes to consider:**\n"
            "- [Cause - brief explanation]\n\n"
            "---\n"
            "**Sources**\n"
            "[1] Filename - Page N\n\n"
            "RULES:\n"
            "- Never invent values. If a number is not in the sources, do not state it.\n"
            "- Be actionable. Engineers need to know what to do.\n"
            "- Do not paste raw source text. Write in your own words.\n"
            "- IMPORTANT: If the sources contain ANY information related to the question, use it. "
            "Only say you cannot find information if the sources contain absolutely nothing relevant. "
            "Regulatory text, standards, and technical specifications ARE valid sources."
            + lang_note
        )
    else:
        style_note = (
            "Use plain language and define technical terms when you first use them."
            if is_beginner
            else "Be precise and technical. Engineers are your audience."
        )
        system_prompt = (
            "You are TorqBase, a precise engineering assistant for two-stroke aircraft engines.\n\n"
            "Using ONLY the numbered sources below, write a clear well-structured answer.\n\n"
            "FORMAT YOUR ANSWER EXACTLY LIKE THIS:\n\n"
            "### [Short title that directly answers the question]\n\n"
            "[Direct answer in 1-2 sentences. State the key fact immediately.]\n\n"
            "[Supporting explanation in clear paragraphs. "
            "Use bullet points for lists of components, steps, or requirements. "
            "Use a table for comparing values or listing multiple specifications.]\n\n"
            "**Key specifications** (ONLY include this section if numeric specs appear in sources):\n"
            "| Parameter | Value | Unit | Source |\n"
            "|-----------|-------|------|--------|\n"
            "[one row per spec - omit this entire table if no specs are present]\n\n"
            "---\n"
            "**Sources**\n"
            "[1] Filename - Page N - one sentence on what it contributed\n"
            "[2] ...\n\n"
            "CRITICAL RULES:\n"
            "1. Write in your own words. Do NOT copy-paste source text into the answer.\n"
            "2. Never include slide markers, page markers, or internal tags in the answer body.\n"
            "3. Never invent or estimate a numeric value. "
            "If a value is not in the sources, say so.\n"
            "4. If sources give different values for the same parameter, show both and flag it.\n"
            "5. Never end mid-sentence. Write a complete answer.\n"
            f"6. {style_note}\n"
            "7. IMPORTANT: If the sources contain ANY information related to the question, "
            "you MUST use it to write an answer. Only say you cannot find information "
            "if the sources contain absolutely nothing relevant - not even indirectly. "
            "Regulatory text, standards, and technical specifications ARE valid sources. "
            "A source about '§ 33.49 Endurance test' IS relevant to 'what is an endurance test'."
            + lang_note
        )

    if state.get("gap_warning"):
        system_prompt += f"\n\n{state['gap_warning']}"
    if state.get("conflicts"):
        system_prompt += _format_conflict_section(list(state.get("conflicts", [])))

    full_system = system_prompt + f"\n\n===SOURCES===\n\n{context}"

    answer_text = chat(
        [
            {"role": "system", "content": full_system},
            {"role": "user", "content": state["question"]},
        ],
        max_tokens=2000,
        temperature=0.1,
    )

    return {"draft": _clean_answer_for_user(answer_text), "citations": citations}


def verify(state: AgentState) -> dict[str, Any]:
    """Grounding check: every numeric claim in draft must appear in a source.

    Also generates 2â€“3 related follow-up questions while we have the context loaded.
    Logs a gap if the answer is ungrounded and we've exhausted retries.
    """
    from config import get_settings
    from agent import verifier as v

    context: list[dict[str, Any]] = []
    for item in state["scratch"]:
        context.extend(item.get("results", []))

    if not get_settings().verify_enabled or not _VERIFY_NUMERIC_RE.search(state["draft"] or ""):
        grounded = True
    else:
        grounded = v.is_grounded(state["draft"], context)

    if not grounded and state["loops"] >= _MAX_LOOPS:
        v.log_gap(state["question"], "weak evidence: verifier could not ground all numeric claims")

    return {"grounded": grounded, "related": []}

