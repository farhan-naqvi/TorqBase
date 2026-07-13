# Hirth Engines Challenge — Project Plan (v2: Agentic + Cloud)
### "TwoStrokeGPT": A self-improving, agentic two-stroke knowledge platform

**Timeframe:** 24–48h hackathon · **Team:** small (2–4) · **Goal:** working demo that wins judges

---

## 1. The idea in one sentence

A **cloud knowledge platform** where anyone can upload two-stroke sources (Hirth manuals, tuning books, forum text) and ask questions — answered by an **agentic LangGraph + RAG system that checks its own answers against the sources before replying**, and that **continuously improves through a human-in-the-loop feedback loop** (not by retraining a model — see §4).

**Why this wins:** it solves the brief's pain ("loss of expert knowledge", "repeated mistakes") with three things plain ChatGPT can't do — answers cited to real manuals, self-correction that cuts hallucinations, and a system that demonstrably gets better as people use it.

---

## 2. Honest framing of "the model learns from mistakes" — READ THIS

This is the single most important thing to get right, because a technical judge **will** probe it.

**The base LLM does not learn or update its weights in this system.** Real model learning needs labelled data, GPU time, and eval infrastructure you won't have in 48h. Do **not** claim "our model retrains itself."

**What genuinely improves over time (and is honest + impressive):**

- **Feedback-weighted retrieval** — 👍/👎 on each answer. Downvoted source chunks get demoted in ranking; upvoted ones get boosted. Retrieval quality improves with use.
- **Correction capture** — when a user or expert corrects an answer, the correction is stored as a *new, high-priority knowledge chunk*. Next time the same question is asked, the corrected answer surfaces. This looks and feels like learning — and it's real.
- **Knowledge-gap detection** — questions that retrieve nothing useful are logged as "gaps" and surfaced to experts to fill.

**Pitch it as:** *"a continuously improving knowledge system with a human-in-the-loop feedback loop"* — never *"a model that learns."*

**Roadmap line (say this if asked about real learning):** "Long-term, the preference data we collect (👍/👎 + corrections) becomes a labelled dataset for periodic fine-tuning or a DPO pass — but that's a production step, not a live demo claim."

---

## 3. Recommended tech stack (cloud-first, agentic)

| Layer | Choice | Why |
|---|---|---|
| Cloud storage + DB + auth | **Supabase** | Postgres + **pgvector** + file storage + auth in ONE product. Users upload files, embeddings + relational data + login all live together. Lowest friction for a hackathon. |
| Upload / ingestion | Supabase Storage → trigger | Drag-drop a PDF → auto parse → chunk → embed → searchable in seconds. |
| Agent orchestration | **LangGraph** | A graph of steps with state + loops. Enables self-correction, routing, and tool use (see §5). |
| Retrieval | LangChain retrievers over pgvector | Plays natively with LangGraph. |
| Embeddings + LLM | Hosted API (OpenAI/Anthropic/Mistral) for demo reliability | Keep Ollama local model as a "data-sovereign / offline" talking point. |
| Frontend | **Next.js + Tailwind** (or Streamlit if backend-only) | Drag-drop upload, chat, feedback buttons, gap view. |
| Knowledge graph | Tables in Supabase + react-force-graph viz | Neo4j only if a graph specialist has spare time. |

**Cloud storage alternatives** (if not Supabase): AWS **S3 + RDS/pgvector** (most "enterprise-credible", more setup); **Cloudflare R2** (no egress fees, cheap); **Pinecone / Qdrant Cloud** (managed vectors only — adds a second system, skip unless you outgrow pgvector).

---

## 4. The feedback loop (your "self-improving" story)

```
User asks → agent answers (cited) → user reacts:
  👍  → boost those source chunks' retrieval weight
  👎  → demote chunks; log as a potential gap
  ✍️ correction → store as high-priority knowledge chunk (expert-verified flag)
        ↳ resurfaces first next time the question is asked
Gaps dashboard → experts fill missing knowledge → base grows
```

No model retraining. The *knowledge base and retrieval ranking* get smarter. Honest, demoable, and directly answers "how does it learn from mistakes?"

---

## 5. Agentic architecture — why LangGraph, not plain RAG

Plain RAG = one call: retrieve → answer. LangGraph gives you a **graph with state, branching, and loops**, which unlocks:

- **Self-correcting RAG (CRAG)** — agent retrieves → drafts answer → a **grader node checks the answer is actually grounded in the retrieved sources** → if not, it re-queries with a refined search before replying. *This is the headline feature: it directly cuts the hallucinations that destroy trust in technical answers.*
- **Routing** — classify the question: factual manual lookup → vector search; diagnostic ("won't start") → knowledge graph (symptom→fix); calculation → tool. Each path is optimized.
- **Tool use** — torque-spec lookup, unit converter, graph queries as callable tools.
- **Feedback state** — the loop in §4 lives naturally as graph state.

**Critical sequencing for 48h:** build **plain RAG first** as the MVP (must work flawlessly). Then add LangGraph's grader/self-correction node on top as differentiation. **If LangGraph breaks at hour 30, fall back to the plain RAG that already works.** Never let the agent be a single point of demo failure.

---

## 6. Maps to the judging criteria

| What they asked for | What you build |
|---|---|
| Prototype knowledge DB/platform | Supabase pgvector store, cloud-hosted |
| AI functions (search, chatbot, recommendation) | Agentic self-correcting RAG chatbot + semantic search + "related issues" |
| Data collection/structuring/maintenance | Cloud upload pipeline + feedback loop + gap dashboard |
| Intuitive UI | Drag-drop upload + chat with citations + feedback buttons |
| (Bonus) Real-world data | Real Hirth manuals + Jennings/Bell books + forum text |
| (Bonus) Tagging / knowledge graph | Engine→Part→Symptom→Fix graph |
| (Bonus) Personalisation / ML | Beginner/Expert mode + feedback-weighted retrieval |

---

## 7. 48-hour timeline

**Hour 0–3 — Setup:** Supabase project (DB + storage + auth), repo + folder structure, lock MVP scope, assign lanes (Ingestion / Agent+API / Frontend / Graph+Pitch). Collect 5–10 real source docs.

**Hour 3–10 — Core (make-or-break):** Upload → cloud → parse/chunk/embed into pgvector. **Plain** RAG `/ask` returning answer + citations. Frontend: upload widget + chat that renders citations. Get ONE question answering well end-to-end ASAP.

**Hour 10–18 — Agentic layer:** Wrap RAG in LangGraph: add the **grader/self-correction node**. Add routing (factual vs diagnostic). Beginner/Expert toggle.

**Hour 18–28 — Feedback + graph (differentiation):** 👍/👎 + correction capture → weighted retrieval. Gap dashboard. Engine→Part→Symptom→Fix tables + graph viz.

**Hour 28–40 — Harden + seed:** Ingest enough data that any judge question returns something good. Loading/empty/error states. Pre-script 3 demo questions that shine. Cache demo answers as a backup.

**Hour 40–48 — Pitch:** Slides (problem → solution → live demo → architecture → honest feedback-loop story → roadmap). Rehearse demo twice. Record a backup video.

---

## 8. MVP vs stretch (cut line)

**Must-have (MVP):** cloud upload → pgvector; **plain** cited RAG chatbot; semantic search; clean UI.
**Should-have (differentiation):** LangGraph self-correction node; feedback loop (👍/👎 + corrections); knowledge graph view.
**Nice-to-have:** routing, gap dashboard, recommendations, Beginner/Expert mode, offline LLM toggle.

> At hour 30 with a shaky MVP: **stop adding features, polish the MVP.** A flawless cited-answer + working-feedback demo beats five half-built agent nodes.

---

## 9. Demo script

1. **Hook (15s):** "Two-stroke expertise is dying in PDFs and forums. We make it uploadable, answerable with sources, and self-improving."
2. **Upload live:** drag a Hirth manual PDF in → "it's now part of the knowledge base."
3. **Ask a real question** → cited answer pointing to the actual manual page.
4. **Show self-correction:** (if built) ask something tricky; mention the grader re-queried before answering — "it checks itself against the sources."
5. **Give feedback:** 👎 + a correction → ask again → corrected answer surfaces first. "It learns from us."
6. **Knowledge graph + gaps dashboard:** show structure + what's missing.
7. **Close:** architecture + honest roadmap (fine-tuning, Hirth partnership, more sources).

---

## 10. Folder structure

```
hirth-twostroke-kb/
├── README.md
├── docker-compose.yml          # local supabase / postgres for dev
├── data/                       # raw source docs
│   ├── manuals/  books/  forums/
├── ingestion/                  # parse → chunk → embed → load (triggered on upload)
│   ├── parse.py  chunk.py  embed.py  load.py
├── api/                        # FastAPI
│   ├── main.py
│   ├── agent/                  # LangGraph
│   │   ├── graph.py            # nodes + edges
│   │   ├── nodes.py            # route, retrieve, draft, grade, rewrite
│   │   └── tools.py
│   ├── feedback.py             # 👍/👎, corrections, weighting
│   └── models.py               # Engine/Part/Symptom/Fix + Feedback schema
├── web/                        # Next.js
│   └── pages/ upload, chat, search, graph, gaps
├── graph/                      # knowledge-graph build + viz data
└── pitch/                      # slides, demo script, backup video
```

---

## 11. Risks & mitigations

- **Agent complexity/latency eats your demo** → plain RAG is the fallback; agent is additive only.
- **PDF parsing is slow** → test PyMuPDF/`unstructured` on real manuals in hour 0–3.
- **LLM/API flakiness live** → cache scripted demo answers + backup video.
- **"Is this just ChatGPT?"** → citations to real manuals + self-correction + the visible feedback loop.
- **"How does it actually learn?"** → answer with §2/§4, never claim model retraining. This is your credibility moment — nail it.

---

---

## 12. How the agent loop actually works (model-driven function calling)

This is the part a technical judge will want explained. The "agent" is not hard-coded if/else logic — **the LLM itself decides what to do**, using function (tool) calling. (See `Hirth_Agent_Loop.svg`.)

The loop:

1. **Input** — the user question is sent to the LLM along with a system prompt *and the JSON schemas of the available tools* (`vector_search`, `graph_lookup`, `spec_lookup`, `unit_convert`).
2. **Reason** — the LLM reads the context and decides: do I have enough to answer, or do I need a tool?
3. **Act** — if it needs a tool, it emits a structured tool call (e.g. `vector_search("piston seizure causes Hirth 3503")`). The runtime executes it.
4. **Observe** — the tool's result (with source references) is appended to the conversation as an "observation."
5. **Loop** — the LLM re-reasons with the new info. It may call more tools (e.g. retrieve, then look up a torque spec). It keeps going until it has what it needs. **The model controls how many steps to take and when to stop** — that's what makes it agentic.
6. **Ground** — before replying, a grading step checks the draft answer is actually supported by the retrieved sources. If not grounded, it loops back and re-queries (corrective RAG).
7. **Answer** — returns a cited answer with source links and a confidence indicator.
8. **Feedback** — user reactions feed the §4 loop.

In LangGraph terms each of these is a **node**; the edges (including the conditional "need a tool?" and "grounded?" branches and the loop-backs) are the graph. State (question, retrieved docs, draft, tool history) flows through it.

**One-line pitch:** "The model isn't just answering — it's deciding which sources to consult, checking its own work against them, and only then replying."

---

## 13. The X-Factor — make it unmistakably *Hirth*

Generic two-stroke Q&A is fine. What wins is showing you understand **Hirth's actual business**: they're a 90-year aviation company and a 30-year UAV/drone propulsion leader — heavy-fuel, NATO-compliant, military + civilian, hybrid propulsion (2025 ePropelled partnership), customers from Israel to China to Africa. Tailor to *that*, not to hobbyist go-karts.

Ranked by impact-vs-effort for the hackathon:

**1. "Tribal knowledge capture" — the emotional hook (high impact, low effort).**
Frame the whole project around Hirth's real risk: 90+ years of expertise sitting in retiring engineers' heads and scattered docs. Your system *captures it before it's lost* — exactly the brief's "continuous loss of expert knowledge." This is a narrative X-factor: costs nothing to build, huge in the pitch.

**2. Predictive / telemetry-aware diagnostics (high impact, medium effort).**
Hirth's 41-series already runs closed-loop control with EGT/RPM/altitude telemetry. Let a user paste an engine log or symptoms ("EGT spiking at altitude, 3503") → the agent correlates against the knowledge graph (symptom→cause→fix) and manual specs. This turns a Q&A bot into a *diagnostic copilot* — far more impressive than search.

**3. Data-sovereign / offline mode for defense (high impact, low effort — it's a toggle + talking point).**
Military/heavy-fuel customers can't send sensitive engine data to a US cloud LLM. Show a local-LLM (Ollama) toggle: "runs fully on-premise, no data leaves the facility." Defense-adjacent judges care about this enormously, and it's cheap to demo.

**4. Multilingual support (medium impact, low effort).**
Global customer base (DE/EN + more). Ask a question in German, get a cited answer — modern LLMs do this for free. Easy win that signals you know the customer map.

**5. Distributor / field-service copilot (medium impact, narrative).**
Hirth sells through a global distributor network whose staff aren't all deep engine experts. Position a "mode" for distributors: guided diagnostics + part lookup so they support customers without flying in a Hirth engineer.

**6. Hybrid-propulsion knowledge frontier (medium impact, forward-looking).**
The ePropelled hybrid line is *so new there's barely any documentation yet*. Pitch your contribution loop as the system that captures this emerging knowledge *as engineers create it* — turning your platform into Hirth's institutional memory for its newest product line.

**7. Visual/diagram intelligence (high impact, higher effort — stretch only).**
Manuals are full of exploded parts diagrams. Multimodal lookup ("which part is #14 in the 3503 crankcase diagram?") is a wow-demo, but costs time. Only if ahead of schedule.

**Recommended X-factor combo for the demo:** lead the *narrative* with #1 (tribal knowledge) + #3 (data sovereignty), and build *one* live wow feature — #2 (telemetry diagnostics) — to prove technical depth. That trio reads as "these people actually understand who Hirth is."

---

## 14. Evidence & sources (cite these when challenged)

Each suggestion above is grounded in a real source. Use these in the pitch and the appendix slide so claims hold up under questioning.

**Hirth company facts (basis for the §13 X-factor):**

- *90+ years aviation, 30 years UAV/drone propulsion; civilian + military proven* — [Hirth Engines on Unmanned Systems Technology](https://www.unmannedsystemstechnology.com/company/hirth-engines/); [Hirth — About Us](https://hirthengines.com/company/).
- *Heavy-fuel, NATO-compliant engines (35HF/3507) for unmanned/maritime/military* — [Hirth 3507 heavy-fuel engine](https://www.unmannedsystemstechnology.com/feature/hirth-engines-3507-heavy-fuel-engine-for-unmanned-maritime-applications/).
- *Closed-loop control + automatic altitude/temperature compensation (basis for telemetry diagnostics, X-factor #2)* — [Hirth 41-series / 2301 feature](https://www.unmannedsystemstechnology.com/feature/hirth-2301-air-cooled-two-stroke-engine-for-uav-platforms/).
- *2025 ePropelled hybrid-propulsion partnership (basis for X-factor #6)* — [Hirth + ePropelled hybrid propulsion](https://insideunmannedsystems.com/hirth-engines-redefining-two-stroke-and-hybrid-propulsion-for-uavs/).
- *Global customer base incl. US, Israel, China, Africa (basis for multilingual + distributor X-factors)* — [Hirth powering UAVs far and wide](https://insideunmannedsystems.com/hirth-engines-powering-uavs-far-and-wide/).

**Real data sources to ingest (basis for §5 corpus):**

- *Hirth official manuals/brochures, free* — [hirthengines.com](https://hirthengines.com/); [Hirth 2704 service manual](https://www.manualslib.com/manual/1190842/Hirth-2704.html).
- *Hirth manual catalogs / parts docs* — [aircraft-reports.com](https://www.aircraft-reports.com/hirth-motoren/); [vintagesnow.com](https://vintagesnow.com/Engine_Manuals.html).
- *Canonical tuning books* — [Graham Bell, Two-Stroke Performance Tuning (PDF)](http://kawatriple.com/manuals/bell/performance-tuning-graham-bell.pdf); Gordon Jennings, *Two-Stroke Tuner's Handbook*.
- *Technical forum knowledge* — [Speed-Talk](https://www.speed-talk.com/); [pit-lane.biz](https://pit-lane.biz); [Bike Chat Forums](https://www.bikechatforums.com/).

**Tooling / architecture choices (basis for §3 stack):**

- *LlamaIndex/LangChain for RAG orchestration; pgvector best when <5–10M vectors on Postgres* — [Best open-source RAG frameworks 2026 (Firecrawl)](https://www.firecrawl.dev/blog/best-open-source-rag-frameworks); [Best vector databases 2026 (Firecrawl)](https://www.firecrawl.dev/blog/best-vector-databases).
- *LangGraph for stateful, looping agent workflows (basis for §5/§12)* — see LangGraph corrective-RAG (CRAG) patterns in the LangChain/LangGraph docs.

**One honesty caveat for the pitch:** the corrective-RAG / self-grading approach reduces but does not eliminate hallucination; present it as a mitigation, not a guarantee. Likewise, the feedback loop improves retrieval and the knowledge base — it does not retrain the base model (see §2).

---

## 15. Memory model — yes, and it's layered (see `Hirth_Data_Flow_Memory.svg`)

"Does it have memory?" — yes, but "memory" is really **five layers with different lifespans**. Conflating them is a common mistake; separating them is a sign you know what you're doing.

1. **Working memory (ephemeral)** — the agent's scratchpad *inside one loop*: tool results, intermediate reasoning, the draft. Lives only for that single question, then discarded. This is just the LangGraph state object.
2. **Conversation memory (session)** — the turns of the current chat. Lets the user ask follow-ups: "and what's the torque for that bolt?" works because the prior turn is remembered. Stored per session.
3. **User profile memory (long-term, per user)** — expertise level (beginner/expert), **language preference (de/en)**, engines they own, recurring questions. Lets answers personalize over time. A row per user in Postgres.
4. **Knowledge base (long-term, shared)** — the pgvector chunks + the Engine→Part→Symptom→Fix graph. This is the system's *semantic memory* — "what it knows" about two-stroke engines. Shared across all users.
5. **Feedback memory (long-term, shared)** — votes, expert corrections, and the gap log. This is the memory that makes the system *improve* (the §4 loop) — corrections resurface, downvoted chunks get demoted.

**For the 48h MVP:** layers 1, 2, and 4 are essentially free (LangGraph state + chat history + your vector store). Layers 3 and 5 are the differentiation — even a minimal version (store language pref + expertise; store corrections) is a strong demo. Don't over-engineer; a few Postgres tables cover all of this.

**Honest note:** none of these layers "retrain the model." They change *what context the model sees* on each call. That's how a stateless LLM produces a system that feels like it remembers and learns (see §2).

## 16. Language handling — English *or* German (and beyond)

The system is multilingual by design, which matters for Hirth's global customer base (§13). Three pieces:

- **Detect** the question's language at ingress (cheap — a small classifier or the LLM itself), and store the user's preference in profile memory (layer 3).
- **Retrieve across languages.** Use a **multilingual embedding model** so a German question can match English manual text and vice-versa. The query "Warum überhitzt..." retrieves the relevant English chunk from a Hirth manual because they land near each other in vector space. This is the key technical point — you do **not** need separate per-language indexes.
- **Answer in the user's language**, with citations pointing at the original source (even if the source is in the other language). The diagram traces exactly this: a German question → English+German sources retrieved → German answer with a cited page reference.

**Demo move:** ask the same question once in English and once in German and show both return the same grounded answer in the right language, citing the same manual page. Cheap to build, and it visibly proves the "works for our global customers" story.

---

## 17. Data sources & ingestion scope — uploaded documents only (current phase)

**Decision:** for now the knowledge comes **only from documents users upload** — no web/forum scraping. Sources can be *unstructured data in any format*. This is a good scope for the hackathon, and you should pitch it as a deliberate choice, not a limitation.

**Why this is the right call:**

- **Clean legal/ethics story.** No copyright or forum-ToS questions — the user owns/controls what they upload. (Big plus for a defense-adjacent company like Hirth.)
- **Data sovereignty fits.** Combined with the offline-LLM option (X-factor #3), "your documents, your infrastructure, nothing leaves" becomes a coherent pitch.
- **Tighter, more reliable demo.** You control the corpus, so retrieval always looks sharp.

**Real data profile (from a sample of actual files).** The corpus is **German-first engineering reference material** — calculation spreadsheets, a diploma thesis, a simulation report, an FAA airworthiness clause, manufacturing tolerances, plus a supplier link. It is **numbers-heavy** and includes **legacy and non-document formats**. Sample: `Berechnung Schallgeschwindigkeit im Auspuff.xlsx`, `Fuel_Kraftstoffe_Übersicht_Daten.xlsx`, `Diplomarbeit Auslegung und Optimierung.pdf`, `Simulation_Modelling.pdf`, `FAR33.49.pdf`, `Mögliche Werkzeugradien.doc`, `HYDAC International.url`. Three consequences: (1) **grounding numbers is the top priority** — never let the LLM invent a value; (2) **spreadsheets are first-class**, not stretch; (3) **German + legacy/edge formats** must be handled from day one.

**"Any format" → what the ingestion pipeline must handle.** This is where the real engineering is. A `format_router` sends each upload to a handler, then all converge into normalize → chunk → embed:

| Uploaded format | Handling | Tool | Sample file |
|---|---|---|---|
| PDF (digital text) | extract text + layout; section/clause metadata | PyMuPDF / `unstructured` | `Diplomarbeit...pdf`, `FAR33.49.pdf`, `Simulation_Modelling.pdf` |
| PDF (scanned / image-only) | **OCR** (de+en) to get text; flag low-confidence | Tesseract / `unstructured` OCR | older scanned manuals |
| Excel / .csv | **table-aware**: keep rows/cols/units/formulas; write to **structured-facts store** for exact-value lookup | pandas / openpyxl | `Berechnung Schallgeschwindigkeit...xlsx`, `Fuel_Kraftstoffe...xlsx` |
| Word / .docx | extract text + tables | python-docx / `unstructured` | modern Word docs |
| **Legacy Word / .doc** | **convert first** (LibreOffice/antiword), then parse — python-docx can't read it | LibreOffice / antiword / textract | `Mögliche Werkzeugradien.doc` |
| PowerPoint / .pptx | slide text + speaker notes | python-pptx | decks |
| Plain text / .md / .html | parse directly | built-in / BeautifulSoup | misc |
| Images (diagrams, photos) | OCR text + optional vision captioning | Tesseract + multimodal LLM | exploded diagrams |
| **URL / .url shortcut** | **not a document** — extract target URL, store as reference link only; **no fetch** under current scope | custom `link_handler` | `HYDAC International.url` |
| Email / .eml | extract body + attachments (recurse) | mailparser | misc |

**Unified pipeline:** every format is converted to **clean text + metadata** (source filename, page/slide, type, language, uploader), then the *same* chunk → embed → store path runs regardless of origin. One normalization layer, many parsers. The recommended `unstructured` library handles most of these in one API, which saves hours.

**MVP cut:** support **PDF (digital) + .docx + .txt** flawlessly first — that covers most manuals. Add **OCR for scanned PDFs** next (Hirth's older manuals are likely scans, so this is high-value). Images/Excel/PPT are stretch.

**Edge cases to handle (and mention to judges):** scanned docs needing OCR, tables that lose meaning when flattened, multi-column manual layouts, mixed-language documents, and very large files. Showing you've thought about messy real-world uploads reads as maturity.

**Demo impact:** the opening move is now even stronger — drag in a Hirth manual PDF (or a scanned one) live, watch it become answerable in seconds. The system is empty until *they* fill it, which makes the "your knowledge, structured" story tangible.

**Roadmap line:** "External sources (forums, web manuals, telemetry feeds) are a phase-2 connector layer — the ingestion pipeline is source-agnostic, so adding them later is additive, not a rebuild."

---

---

## 18. Retrieval & UX enhancements (vetted add-ons)

Each was reasoned through for impact-vs-effort before inclusion. Three are substance, two are demo polish, one is roadmap.

**Substance (build these):**

- **Re-ranking layer** — retrieval returns top-20 candidates; a cross-encoder (`ms-marco-MiniLM-L6`, small/CPU-friendly) re-scores them against the exact query and passes the top-5 to the agent. Vector/hybrid search finds *related* chunks; the re-ranker finds the *best* ones — a measurable precision jump, and especially valuable for this numeric corpus where the right spec chunk matters. ~One model, one call.
- **Recommendation system** — *explicitly required by the challenge brief* and was missing. Cheap form: "related questions" generated from retrieved content + "related parts/symptoms" from knowledge-graph neighbours. No bespoke ML needed.
- **Chunk deduplication at ingest** — cosine > 0.98 = duplicate (e.g. a spec copied between a manual and a spreadsheet). Skips the duplicate vector to avoid inflated confidence and wasted context. **Critical caveat:** do not blind-delete — *merge and keep all source references* on the retained chunk, or you break `conflict_check` and provenance, which depend on seeing multiple sources.

**Demo polish (cheap, worth it):**

- **Streaming responses** — stream tokens as they generate instead of a multi-second freeze. Pure perceived-quality win in a live demo. One wrinkle: stream *after* the grounding check passes (or show "checking sources…" then stream), since you can't emit a final answer before it's verified. ~An afternoon.
- **Progressive upload feedback** — show pipeline progress (Parsing… Chunking… Embedding… N chunks indexed) so a judge doesn't think the system froze during the opening upload demo. SSE/WebSocket, low effort.

**Roadmap (talk about, don't build for the hackathon):**

- **Document versioning** — re-uploading the same filename → "replace or add as new version?"; replace deletes old vectors by `doc_id` and re-ingests, preventing stale-chunk false-conflicts. Genuinely important for a real Hirth deployment, irrelevant to a 5-minute demo. Pitch it as designed-for; build only if ahead of schedule.

- **Figure & diagram intelligence** (= X-factor #7, now spec'd) — added as an end-of-pipeline stretch branch (`figure_handler.py`). Pragmatic approach, *not* full computer vision: (1) extract embedded figures from PDFs (PyMuPDF) → image store; (2) vision-LLM **caption** each figure so the diagram becomes searchable text in the normal chunk→embed path; (3) OCR callout numbers and **link them to the parts list / Engine→Part graph**; (4) `source_viewer` returns the **actual image** beside the cited answer. Gets ~80% of the wow ("ask about a diagram, get an answer + the picture") for exploded parts diagrams, schematics, and the charts in the thesis/simulation PDFs. True spatial "point at the exact bolt" QA is out of scope. **Build only if MVP + the substance add-ons (§18) are solid** — captioning every figure costs ingest time and vision-LLM calls.

**Honest priority note:** the re-ranker and recommendation system are the two that change outcomes (precision + a scored requirement). Streaming/progress make the demo *feel* better but add no intelligence. Versioning is product maturity, not hackathon scope.

---

*Sources cited above (Hirth facts, tooling) were gathered via web research during planning and are also listed in the accompanying chat messages. Note: per the scope decision in §17, the live system's knowledge base is populated from user uploads, not from these web sources — those informed the plan, not the corpus.*
