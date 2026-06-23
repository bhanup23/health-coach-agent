# Technical Report: Health Coach AI Agent

**Project:** Health Coach AI Agent  
**Author:** Bhanu Pratap  
**Date:** June 2025  
**Context:** Stealth AI  Internship Assignment  
**Repository:** [https://github.com/bhanup23/health-coach-agent](https://github.com/bhanup23/health-coach-agent)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Requirements Analysis](#2-requirements-analysis)
3. [Architecture Design](#3-architecture-design)
4. [Extraction Pipeline](#4-extraction-pipeline)
5. [RAG Pipeline](#5-rag-pipeline)
6. [Agent Design](#6-agent-design)
7. [Memory Management](#7-memory-management)
8. [Hallucination Prevention](#8-hallucination-prevention)
9. [Tradeoffs](#9-tradeoffs)
10. [Future Improvements](#10-future-improvements)
11. [Challenges Encountered](#11-challenges-encountered)

---

## 1. Problem Statement

### 1.1 Context

Wellness coaching programmes typically begin with a patient onboarding process in which practitioners capture background information about the individual — their age, wellness goals, sleep patterns, current struggles, and lifestyle habits. In practice, this information is rarely structured. It is entered as freeform notes, varies significantly in completeness and phrasing, and is difficult to process programmatically without manual review.

Once onboarded, patients interact with their coach to receive guidance aligned with a specific wellness protocol — a structured programme that defines day-by-day activities, nutrition recommendations, sleep hygiene practices, and habit formation strategies. In a human-led coaching model, the coach must recall the patient's profile, track their progress through the protocol, and tailor responses accordingly. This is cognitively demanding and difficult to scale.

### 1.2 Core Problems

This project addresses two distinct but related problems:

**Problem 1 — Unstructured data intake.** Patient onboarding notes are freeform, inconsistent, and lack a schema. Downstream systems that need to personalise coaching responses cannot reliably use this data without first converting it into a structured, typed representation.

**Problem 2 — Hallucination risk in health-adjacent AI.** General-purpose LLMs will attempt to answer any question, including health-related queries, using parametric knowledge that may be outdated, inaccurate, or inconsistent with the specific protocol a patient is following. In wellness coaching, this risk is unacceptable: the agent should answer strictly from the verified protocol, not from general training data.

### 1.3 Scope

This system is scoped to:
- Single-user, single-session wellness coaching interactions.
- A single wellness protocol document as the exclusive knowledge source.
- Wellness guidance only — not medical diagnosis, clinical decision-making, or emergency advice.

---

## 2. Requirements Analysis

### 2.1 Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-01 | Accept unstructured patient onboarding text as input |
| FR-02 | Extract a typed, validated patient profile from the onboarding text |
| FR-03 | Store the patient profile and make it available throughout the session |
| FR-04 | Accept natural language queries from the user in a chat interface |
| FR-05 | Retrieve relevant protocol context before generating any response |
| FR-06 | Generate responses grounded exclusively in retrieved protocol context |
| FR-07 | Refuse to answer questions outside the protocol knowledge base |
| FR-08 | Display the retrieved source chunks alongside each response |
| FR-09 | Maintain multi-turn conversation history within a session |
| FR-10 | Adapt coaching responses to the patient's current protocol day |

### 2.2 Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | Modular codebase with clear separation of concerns across components |
| NFR-02 | Secure API key management via environment variables |
| NFR-03 | No external infrastructure dependencies beyond the Gemini API |
| NFR-04 | Interpretable refusal behaviour — no silent hallucination |
| NFR-05 | Deployable via Streamlit Community Cloud without additional configuration |

### 2.3 Out of Scope

The following are explicitly out of scope for this implementation:

- Multi-user support with isolated profiles
- Cross-session memory persistence
- Automated RAG quality evaluation
- Production logging, monitoring, or alerting
- Medical diagnosis or clinical guidance

---

## 3. Architecture Design

### 3.1 Overview

The system is organised into five loosely coupled layers, each with a single responsibility:

```
┌─────────────────────────────────┐
│         Presentation Layer      │  app.py — Streamlit UI, session state
├─────────────────────────────────┤
│          Agent Layer            │  agents/health_agent.py — prompt assembly, grounded response
├─────────────────────────────────┤
│        Retrieval Layer          │  rag/retriever.py — FAISS search, HuggingFace embeddings
├─────────────────────────────────┤
│        Extraction Layer         │  parser/profile_parser.py — Pydantic structured extraction
├─────────────────────────────────┤
│     Configuration & Prompts     │  utils/config.py, prompts/system_prompt.py
└─────────────────────────────────┘
```

Each layer communicates through explicit function calls and typed data structures. No layer holds persistent state directly — state flows through Streamlit's session state, which acts as the runtime data store for the session.

### 3.2 Component Responsibilities

**`app.py` (Presentation Layer)**  
Entry point and UI orchestrator. Responsible for rendering the onboarding interface, triggering profile extraction on first submission, maintaining the chat loop, passing user messages to the agent, and displaying responses alongside retrieved source chunks. All session state (`st.session_state`) is managed here.

**`agents/health_agent.py` (Agent Layer)**  
Receives the patient profile, current protocol day, conversation history, and the user's message. Calls the retriever to fetch relevant protocol chunks, assembles the full context-grounded response using the system prompt template, and returns the response text to the presentation layer. The implementation prioritises deterministic, protocol-grounded behaviour using retrieved context directly.

**`rag/retriever.py` (Retrieval Layer)**  
On initialisation, loads `data/protocol.pdf`, splits it into chunks using LangChain's `RecursiveCharacterTextSplitter`, generates embeddings using the HuggingFace Sentence-Transformers model (`all-MiniLM-L6-v2`), and builds a FAISS index. At query time, embeds the user's query using the same model and returns the top-K most similar chunks.

**`parser/profile_parser.py` (Extraction Layer)**  
Accepts raw onboarding text and converts it into a validated `PatientProfile` object using a lightweight local extraction pipeline combined with Pydantic validation. Handles graceful fallback for missing or ambiguous fields, ensuring downstream components always receive a typed, valid profile.

**`prompts/system_prompt.py` (Prompt Templates)**  
Defines the system prompt template. The prompt injects the patient profile, current protocol day, retrieved RAG context, and conversation history. It also contains the grounding and refusal instructions that govern agent behaviour.

**`utils/config.py` (Configuration)**  
Loads environment variables, exposes model names, chunk size, overlap, and top-K retrieval count as typed constants. Provides a single source of truth for all configuration values.

### 3.3 Data Flow

```
[1] User submits onboarding text
        ↓
[2] profile_parser.py → local extraction + Pydantic validation
        ↓
[3] PatientProfile (Pydantic) stored in st.session_state
        ↓
[4] User submits chat query
        ↓
[5] retriever.py → HuggingFace embeddings → FAISS → top-K chunks
        ↓
[6] health_agent.py assembles grounded response:
      system_prompt + patient_profile + protocol_day
      + conversation_history + retrieved_chunks + user_query
        ↓
[7] Context-grounded response assembled and returned
        ↓
[8] app.py renders response + source chunks in UI
        ↓
[9] Message appended to conversation history in st.session_state
```

---

## 4. Extraction Pipeline

### 4.1 Problem

Patient onboarding notes arrive as unstructured natural language. The same information may be expressed in dozens of different ways:

- "John, 34, wants to sleep better and lose weight" 
- "Patient is a 34-year-old male. Goals include weight management. Sleep is a challenge — averaging 5 hours."
- "34yr old. Struggles with consistency and late-night eating. Wants energy and to lose some weight."

A downstream coaching agent cannot reliably personalise responses without a structured, typed representation of this data.

### 4.2 Approach: Structured Profile Extraction with Pydantic

The system converts onboarding notes into a structured `PatientProfile` object using a local extraction pipeline combined with Pydantic validation. During development, Gemini structured output was explored as the extraction mechanism. However, practical constraints — including API authentication inconsistencies across environments and quota restrictions during testing — led to a migration toward a lightweight local extraction approach that runs without external API dependencies, ensuring reliable execution across all deployment environments.

**Pydantic Schema:**

```python
class PatientProfile(BaseModel):
    patient_name: str
    age: int
    primary_wellness_goals: list[str]
    sleeping_habits: str
    current_struggles: list[str]
```

The schema enforces runtime type validation on all extracted fields. Pydantic raises explicit validation errors if a field is missing or of the wrong type, which are caught and resolved with sensible defaults rather than allowed to propagate silently.

### 4.3 Why This Approach

Three options were considered across the development lifecycle:

**Option A — Gemini structured output with Pydantic schema binding.**  
Explored during initial development. Provides API-level schema guarantees and eliminates JSON parsing errors. Rejected for the final implementation due to API authentication inconsistencies and quota limitations that introduced unreliability in the extraction step.

**Option B — Prompt the model to return JSON as a string, then parse.**  
Fragile in practice. Models frequently wrap JSON in markdown code fences or add commentary, causing silent parse failures that could propagate incorrect profile data downstream.

**Option C — Lightweight local extraction with Pydantic validation (chosen).**  
Runs entirely without external API dependencies. Pydantic enforces schema conformance at runtime. Fallback defaults are applied for missing fields. This approach prioritises reliability and deployment simplicity, which is the correct trade-off for a demo-stage system where extraction API availability cannot be guaranteed.

### 4.4 Fallback Behaviour

If extraction produces missing or invalid fields, the parser applies typed default values for each field and continues. The UI surfaces a summary of the extracted profile so the user can verify correctness before beginning the coaching session.

---

## 5. RAG Pipeline

### 5.1 Motivation

Rather than relying on Gemini's parametric knowledge to answer wellness coaching questions, all responses are grounded in a specific wellness protocol PDF. This has two benefits: the agent's responses are consistent with the programme the patient is following, and the knowledge boundary is explicit and auditable.

### 5.2 Pipeline Stages

**Stage 1 — Document Loading**  
The wellness protocol PDF is loaded using LangChain's `PyPDFLoader`, which extracts page-level text. This produces a list of `Document` objects, each corresponding to one page.

**Stage 2 — Text Chunking**  
Page-level documents are split into smaller chunks using `RecursiveCharacterTextSplitter`. The recursive splitter attempts to split on paragraph boundaries first, then sentence boundaries, then character boundaries — preserving semantic coherence as much as possible.

Default configuration:
```
CHUNK_SIZE=1000    # characters per chunk
CHUNK_OVERLAP=200  # overlap between adjacent chunks
```

The 200-character overlap ensures that information spanning a chunk boundary is not lost — both adjacent chunks contain the bridging content.

**Stage 3 — Embedding Generation**  
Each chunk is embedded using the Sentence-Transformers model `all-MiniLM-L6-v2` via HuggingFace Embeddings. The model generates dense 384-dimensional semantic vectors that encode meaning rather than lexical content — chunks about "sleep quality" and "rest and recovery" will have similar embeddings even if they share no common words. This model runs locally with no API dependency, eliminating quota constraints and network latency from the retrieval path. During development, the Gemini Embedding model (`models/gemini-embedding-001`) was also evaluated, but `all-MiniLM-L6-v2` was selected for its zero-cost local execution and strong semantic retrieval performance on short to medium text spans.

**Stage 4 — FAISS Index Construction**  
The embeddings are stored in a FAISS (Facebook AI Similarity Search) index, which enables efficient approximate nearest-neighbour search at query time. The index is built once and held in memory for the session.

**Stage 5 — Query-Time Retrieval**  
When the user submits a question, the query is embedded using the same Gemini Embedding model. FAISS returns the top-K chunks whose embeddings are most similar to the query embedding. These chunks are passed to the agent as retrieved context.

Default: `TOP_K_RESULTS=4`

### 5.3 Chunking Strategy Rationale

The choice of chunk size involves a fundamental tradeoff:

- **Smaller chunks** → higher precision (retrieved chunks are tightly focused on the query topic) but lower recall (relevant context may be split across multiple chunks, neither of which scores highly alone).
- **Larger chunks** → higher recall but lower precision (retrieved chunks contain more irrelevant surrounding content, diluting the signal).

A chunk size of 1000 characters was selected as a starting point based on the typical length of a protocol section. The 200-character overlap is set at 20% of chunk size, a common empirical starting point for RAG systems.

---

## 6. Agent Design

### 6.1 Role

The health coaching agent is the system's reasoning core. It receives all available context — the patient profile, the current protocol day, the conversation history, and the retrieved RAG chunks — and produces a response that is personalised, grounded, and assembled deterministically from the retrieved protocol content. The final implementation prioritises protocol-grounded behaviour over open-ended generative responses, ensuring that every answer can be traced back to a specific retrieved chunk from the wellness document.

### 6.2 Prompt Structure

The agent prompt is assembled from five components, in this order:

```
[1] SYSTEM INSTRUCTIONS
    → Role definition (wellness coach)
    → Grounding constraint (answer only from retrieved context)
    → Refusal instruction (explicit fallback if context is insufficient)

[2] PATIENT PROFILE
    → Name, age, wellness goals, sleeping habits, current struggles

[3] CURRENT PROTOCOL DAY
    → Day number injected to enable day-aware responses

[4] RETRIEVED RAG CONTEXT
    → Top-K document chunks from FAISS retrieval
    → Labelled as "Protocol Context" to distinguish from conversation

[5] CONVERSATION HISTORY
    → All prior turns in the session (role: user/assistant alternating)

[6] CURRENT USER MESSAGE
    → The user's most recent query
```

This ordering is deliberate. The system instructions are placed first to establish constraints before the model sees any content. The patient profile follows to personalise the framing. The RAG context is placed immediately before the conversation history so it is in the model's recency window when generating the response.

### 6.3 Day-Aware Coaching

The current protocol day is injected into the prompt as a typed integer. The system prompt instructs the agent to interpret responses in the context of this day — for example, Day 1 guidance should focus on orientation and baseline habits, while Day 14 guidance can reference habits that should already be established.

### 6.4 Grounding Instruction

The system prompt contains an explicit instruction:

> "Answer only using the protocol context provided above. If the answer cannot be found in the provided context, respond with: 'I cannot find that information in the wellness protocol.' Do not use your general knowledge to supplement the response."

This instruction is the primary mechanism for hallucination prevention. See Section 8 for the full strategy.

---

## 7. Memory Management

### 7.1 Approach: Streamlit Session State

Conversation history is maintained using `st.session_state`, Streamlit's built-in key-value store that persists across UI re-renders within a single browser session.

Two keys are used:

```python
st.session_state["patient_profile"]    # PatientProfile object
st.session_state["conversation_history"]  # list of {"role": ..., "content": ...} dicts
```

At each turn, the user's message is appended before the agent call, and the agent's response is appended after. The full history is passed to the agent on every turn.

### 7.2 History Format

Conversation history is stored as a list of role-tagged message dicts, compatible with the Gemini API's `contents` format:

```python
[
    {"role": "user",      "content": "What should I focus on for sleep this week?"},
    {"role": "assistant", "content": "Based on the wellness protocol for Day 7..."},
    {"role": "user",      "content": "What about nutrition?"},
]
```

### 7.3 Known Limitation: Session Scope

Session state is cleared when the browser tab is refreshed or closed. This means conversation history and patient profiles do not persist across sessions. This is the most significant architectural limitation of the current implementation.

**Why this decision was made:** Persistence requires a database (SQLite, PostgreSQL) or external store (Redis). For an MVP internship project, introducing a database layer would add infrastructure complexity and deployment friction without changing the core engineering demonstration. The limitation is documented and flagged as a future improvement.

### 7.4 Context Window Growth

As conversation history grows, the total size of the assembled context increases. For long sessions with many turns, this can affect retrieval relevance and response coherence as older turns accumulate. A mitigation strategy — conversation summarisation memory, where older turns are replaced with a rolling LLM-generated summary — is noted in Future Improvements but not implemented in this version.

---

## 8. Hallucination Prevention

### 8.1 Why This Is a Design Priority

In wellness coaching, incorrect information is not merely unhelpful — it may lead a user to make changes to their sleep, nutrition, or exercise habits based on advice that contradicts their specific protocol. The risk of a general-purpose LLM confidently generating plausible but incorrect wellness guidance is a first-class engineering concern.

### 8.2 Strategy

The system uses three complementary mechanisms to prevent hallucination:

**Mechanism 1 — Retrieval-First Generation**

The agent never generates a response without first retrieving context from the protocol document. The retrieved chunks are the primary input to the generation step. The model is explicitly instructed to treat these chunks as its sole knowledge source for the question at hand.

This transforms the agent's task from "answer this wellness question" (open-ended, hallucination-prone) to "summarise and apply these specific protocol excerpts to this patient's situation" (constrained, grounded).

**Mechanism 2 — Explicit Refusal Instruction**

The system prompt contains a hard constraint: if the retrieved context does not contain a sufficient answer, the agent must respond with the fixed refusal message rather than attempting to answer from general knowledge.

This is a deliberate design choice over confidence thresholding (discussed in Section 9). The refusal is visible, auditable, and consistent — a reviewer can see exactly when and why the agent declined to answer.

**Mechanism 3 — Source Transparency**

The UI renders the retrieved chunks alongside each agent response. This allows any user or evaluator to verify that the agent's answer is traceable to the source document. If a response contains information not present in the displayed chunks, that is a signal that the grounding constraint was not honoured — and the UI makes this immediately visible.

### 8.3 What This Does Not Prevent

The system does not prevent the agent from misinterpreting or misapplying retrieved context. If the retrieved chunks contain ambiguous content, the agent may produce a response that is technically grounded but misleading. This is a limitation of all RAG systems without additional verification layers (e.g., faithfulness scoring).

---

## 9. Tradeoffs

### 9.1 FAISS vs. Hosted Vector Database

| Dimension | FAISS | Pinecone / Weaviate |
|-----------|-------|---------------------|
| Infrastructure | None (in-process) | External service required |
| Latency | Sub-millisecond (in-memory) | Network round-trip |
| Scalability | Single machine, single index | Horizontally scalable |
| Cost | Free | Usage-based pricing |
| Index persistence | Must rebuild on restart | Persistent by default |
| Setup complexity | Minimal | Significant |

**Decision:** FAISS. For a single-document knowledge base, the scalability and persistence advantages of a hosted database are irrelevant. The operational simplicity and zero cost of FAISS are decisive for an internship-stage project.

---

### 9.2 Local Extraction vs. API-Based Structured Output (Gemini)

| Dimension | Local Extraction + Pydantic | Gemini Structured Output |
|-----------|----------------------------|--------------------------|
| API dependency | None | Required |
| Reliability in demo | High (no quota/auth issues) | Variable (quota, auth failures) |
| Schema enforcement | Pydantic at runtime | API-level guarantee |
| Parsing errors | N/A | Impossible (when available) |
| Cost | Free | API call per extraction |
| Deployment simplicity | High | Medium |

**Decision:** Local extraction with Pydantic validation. Gemini structured output was the initial design choice and remains the stronger long-term option for production use. It was de-prioritised in the final implementation due to API authentication inconsistencies and quota limitations encountered during development and deployment. The local approach gives equivalent schema enforcement via Pydantic with zero external dependency, which is the right trade-off for a reliable internship demo.

---

### 9.3 HuggingFace Embeddings vs. Gemini Embedding Model

| Dimension | all-MiniLM-L6-v2 (HuggingFace) | Gemini Embedding |
|-----------|--------------------------------|-----------------|
| Execution | Local (no API call) | Remote API call |
| Cost | Free | Per-request pricing |
| Latency | In-process, fast | Network round-trip |
| Embedding quality | Strong for short–medium text | Strong, general-purpose |
| API dependency | None | Google AI API required |
| Quota risk | None | Rate limits apply |

**Decision:** `all-MiniLM-L6-v2` via HuggingFace. The Gemini Embedding model was evaluated during development. Both models produce strong semantic representations for protocol document retrieval. The HuggingFace model was selected for its zero-cost local execution and absence of quota constraints — particularly important for a demo deployment that may experience bursty usage during evaluation.

---

### 9.4 Prompt-Based Refusal vs. Similarity Threshold Filtering

| Dimension | Prompt-Based Refusal | Similarity Threshold |
|-----------|---------------------|---------------------|
| Calibration required | None | Yes (threshold tuning) |
| Interpretability | High (explicit instruction) | Low (numeric cutoff) |
| False positive risk | Low | Moderate |
| False negative risk | Low | Moderate |
| Auditability | Full (visible in prompt) | Partial (hidden in retriever) |
| Implementation effort | Minimal | Moderate |

**Decision:** Prompt-based refusal. For an MVP, interpretability and auditability are more valuable than the marginal precision gain of threshold filtering. A hybrid approach (threshold + prompt instruction) is noted as a future improvement.

---

### 9.5 Streamlit Session State vs. External Memory Store

| Dimension | Session State | Redis / Database |
|-----------|--------------|-----------------|
| Persistence | Session only | Cross-session |
| Setup complexity | Zero | Significant |
| Multi-user support | No | Yes |
| Deployment friction | None | Infrastructure required |
| Suitable for MVP | Yes | Overkill |

**Decision:** Session state. The persistence limitation is documented and accepted for this stage of the project.

---

## 10. Future Improvements

The following improvements are prioritised by expected impact on system quality:

### 10.1 High Impact

**Conversation Summarisation Memory**  
As conversation history grows, the assembled context increases in size, potentially affecting retrieval coherence and response quality in long sessions. Replacing raw history with an LLM-generated rolling summary would cap context growth while preserving the relevant personalisation context.

**LangSmith Tracing and Observability**  
Integrating LangSmith would enable full trace visibility into every retrieval call, prompt assembly step, and generation. This is the most important operational improvement for moving beyond manual testing.

**Automated RAG Evaluation with RAGAS**  
RAGAS provides standardised metrics for RAG system quality: faithfulness (does the response contradict the context?), answer relevance (does the response address the question?), and context precision (were the right chunks retrieved?). Automating this evaluation would replace manual testing with a reproducible benchmark.

### 10.2 Medium Impact

**Similarity Threshold Filtering**  
Adding a configurable similarity threshold to the retriever would catch genuinely out-of-domain queries before they reach the LLM, reducing unnecessary API calls and providing a secondary hallucination prevention layer.

**Persistent Patient Profiles**  
Storing patient profiles and session summaries in SQLite would allow the agent to resume coaching across sessions, which is the most significant UX improvement available.

**Profile Editing Interface**  
Allowing users to review and correct their extracted profile before beginning coaching would increase trust in the extraction pipeline and reduce errors caused by ambiguous onboarding notes.

### 10.3 Lower Impact (Infrastructure / Polish)

- **Streaming responses** — Token-by-token streaming for a more responsive chat experience.
- **Query parameter support** — Pass protocol day and patient name via URL for easier reviewer testing.
- **Docker containerisation** — Reproducible deployment environment.
- **Authentication layer** — Multi-user support with isolated session state.

---

## 11. Challenges Encountered

Development of the Health Coach AI Agent involved several practical engineering challenges that are not visible in the final codebase but are worth documenting as part of an honest account of the project lifecycle.

**Gemini API Authentication Inconsistencies**  
During early development, Gemini API authentication behaved inconsistently across local and Streamlit Cloud environments. Credentials that worked locally failed in deployment due to differences in how environment variables are injected. This was resolved through explicit configuration management in `utils/config.py` and a migration of the embedding and extraction components to locally-executable alternatives that do not require live API access.

**Model Quota and Rate-Limit Restrictions**  
API quota limits on the free tier restricted the number of extraction and embedding calls available during iterative testing. This directly motivated the decision to use `all-MiniLM-L6-v2` for embeddings (zero quota cost, local execution) and the local extraction pipeline for profile parsing.

**Accidental Virtual Environment Commit**  
During early repository setup, the Python virtual environment (`venv/`) was committed to the repository, causing the repository size to balloon significantly. This was resolved through a `git filter-branch` cleanup, a corrected `.gitignore`, and a repository history rewrite.

**Streamlit Deployment Dependency Conflicts**  
The initial `requirements.txt` contained version combinations that produced dependency conflicts on Streamlit Community Cloud's build environment. Resolving this required iterative constraint relaxation and pinning specific package versions compatible with both local Python 3.12 and the cloud runtime.

**Secret Scanning Protections**  
An early commit included a hardcoded API key in a configuration file. GitHub's secret scanning flagged this and blocked the push. The key was rotated, removed from history using `git filter-branch`, and replaced with `.env`-based configuration managed by `python-dotenv`.

These challenges contributed directly to several architectural decisions in the final implementation, including the migration to locally-executable components, the structured configuration module, and the improved `.gitignore` setup.

---

## Summary

The Health Coach AI Agent demonstrates a complete, working AI application pipeline: from unstructured data intake through Pydantic-validated structured extraction, FAISS-backed retrieval-augmented generation with HuggingFace embeddings, and protocol-grounded response assembly. The architectural decisions made throughout — FAISS over hosted databases, local extraction over API-dependent structured output, HuggingFace embeddings over cloud embedding APIs, prompt-based refusal over threshold filtering, session state over persistent storage — reflect a consistent engineering philosophy: prefer reliability, interpretability, and deployment simplicity over premature complexity at the MVP stage.

The most significant known limitation is the lack of cross-session memory persistence. The most significant known risk is the absence of automated RAG evaluation, which means retrieval quality is currently validated only through manual testing. Both are documented, prioritised, and planned as next-step improvements.

---

*Report prepared for internship evaluation purposes. The system described is a wellness coaching assistant and must not be used for medical advice, clinical decision-making, or as a substitute for professional healthcare.*
