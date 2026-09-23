# Materials-Informatics Copilot — Project Handoff

## Current status — 23 September 2026

The ingestion, retrieval, grounded multi-tool agent, evaluation, Streamlit UI,
and Community Cloud deployment milestones are implemented. Start with
[README.md](README.md) for current setup, architecture, limitations, and the live demo.
The sections below preserve the original project plan and may describe older states;
the ingestion milestone is no longer the next task.

- Corpus: 40 HEA arXiv abstracts in `data/hea_papers.json`.
- Missing/empty Chroma indexes initialize automatically before literature search.
- Literature answers validate retrieved IDs and abstract quotations, retry once,
  and build citations from metadata. Valid computed results survive literature abstention.
- Stored 16-case evaluation: 16 tool-selection matches, 7/7 required citation checks,
  zero execution errors. These checks do not establish scientific answer quality.
- Latest verified offline suite after portfolio cleanup: 100 passing tests.
- Live Cloud smoke tests succeeded for the TiO2 band gap and mixed Ni question.
- Cloud secrets must be quoted, root-level TOML strings; `.env` is local-only.
- Portfolio documentation, live screenshots, and safe exception-type logging are
  included. Request user approval before future commits or pushes.

Manage credentials privately in local `.env` and Cloud Secrets. Never copy their
values into docs, logs, chat, tests, or Git.

## 1. Project purpose

This is a portfolio project for AI Engineer / Applied AI / LLM Engineer roles.

The goal is to build a **Materials-Informatics Copilot** that combines:

1. **Structured materials-property lookup** from the Materials Project API.
2. **Literature retrieval (RAG)** over a focused set of high-entropy-alloy papers from arXiv.
3. **LLM tool calling** so the model can decide when to:
   - call a property lookup tool, and/or
   - search the literature.
4. A simple **Streamlit chat interface**.
5. A clean GitHub repository with tests, documentation, screenshots, and an architecture diagram.

The project should be technically credible, beginner-readable, and easy to explain in an interview.

## 2. User background and design rationale

The project owner has:

- a B.Tech in Metallurgical & Materials Engineering
- an MSc in Data Science
- beginner-level hands-on experience with AI engineering projects
- working knowledge of Python, APIs, Git/GitHub, and data analysis

The project intentionally combines **materials science domain knowledge + AI engineering** so it is more distinctive than a generic "chat with PDFs" RAG project.

The code must therefore remain understandable enough for the owner to explain:

- what each component does
- why each technology is used
- how tool calling works
- how retrieval works
- how embeddings work
- how the system is evaluated
- where the limitations are

Do not optimize for cleverness at the expense of clarity.

## 3. Current environment

Local project folder:

```text
/Users/tiashaverse/materials-copilot
```

GitHub repository:

```text
https://github.com/Tiasha-17/materials-co-pilot
```

Branch:

```text
main
```

Python:

```text
Python 3.12.14
```

Virtual environment:

```text
.venv
```

The virtual environment is already created and working.

## 4. Secrets and security

The project uses a local `.env` file.

Current environment variables:

```text
MP_API_KEY=...
GROQ_API_KEY=...
```

Important:

- `.env` must NEVER be committed.
- `.venv/` must NEVER be committed.
- Do not print API keys.
- Do not add secrets to tests, README, example outputs, screenshots, or deployment files.
- Before every Git commit that changes configuration, confirm `.env` is ignored.

Current `.gitignore` includes at least:

```text
.env
.venv/
__pycache__/
*.pyc
.DS_Store
chroma_db/
```

## 5. Current dependencies

Current `requirements.txt` contains:

```text
mp-api
python-dotenv
groq
arxiv
```

Additional dependencies may be added later as needed.

Likely future additions:

```text
chromadb
voyageai
streamlit
pytest
```

Do not add all future dependencies blindly. Add them when their phase starts.

## 6. Current repository structure

At the current handoff point, the repo contains approximately:

```text
materials-copilot/
├── .gitignore
├── .env                  # local only, ignored by Git
├── .venv/                # local only, ignored by Git
├── data/
├── copilot.py
├── ingest_arxiv.py
├── requirements.txt
├── test_arxiv.py
├── test_groq.py
├── test_materials_api.py
└── tools.py
```

`ingest_arxiv.py` was only just started and may contain only the first few lines/constants.

## 7. Completed milestone: Materials Project integration

A working function exists in `tools.py`:

```python
get_material(formula: str)
```

Its current behavior:

1. loads the Materials Project API key from `.env`
2. queries `mpr.materials.summary.search(...)`
3. requests fields including `material_id`, `formula_pretty`, `band_gap`, `energy_above_hull`, and `symmetry`
4. sorts returned structures by `energy_above_hull`
5. selects the lowest-energy-above-hull result
6. returns a clean dictionary
7. returns a friendly error dictionary if no material is found

Example successful output:

```python
{
    "material_id": "mp-390",
    "formula": "TiO2",
    "band_gap": 2.063,
    "energy_above_hull": 0.0,
    "crystal_system": "Tetragonal",
    "space_group": "I4_1/amd"
}
```

An invalid formula test such as `get_material("XYZ123")` returns:

```python
{"error": "No material found for XYZ123"}
```

### Important scientific limitation

The current implementation selects the structure with the lowest `energy_above_hull`.

That is a useful deterministic default, but it must NOT be presented as "the universally correct material record."

For polymorphic materials, multiple structures can have different computed properties. The assistant should be explicit that values are:

- Materials Project computed values
- structure-specific
- not necessarily experimental values

Future improvements may expose multiple structures or allow explicit material IDs/polymorph selection.

Do not silently erase this limitation.

## 8. Completed milestone: Groq LLM integration

Anthropic was originally considered but was not used because the API required paid billing.

The project currently uses the **Groq API free tier**.

Python SDK:

```text
groq
```

Current model:

```text
openai/gpt-oss-120b
```

Important clarification: this model ID is called `openai/gpt-oss-120b`, but it is being accessed through the **Groq API** in this project.

Do not replace Groq unless there is a clear technical reason.

## 9. Completed milestone: local tool calling

`copilot.py` already implements a working local tool-calling flow.

Current flow:

```text
User question
    ↓
Groq LLM
    ↓
LLM decides whether get_material is needed
    ↓
Python parses tool-call JSON
    ↓
get_material(formula)
    ↓
Materials Project API
    ↓
tool result is appended to conversation
    ↓
Groq LLM generates final natural-language answer
```

The tool schema currently exposes `get_material` with a required string argument `formula`.

The code already handles the case where no tool call is returned. It also validates the requested tool name before execution. Tool output is sent back using JSON.

Example tested user question:

```text
What is the band gap of TiO2?
```

Example final answer:

```text
The Materials Project reports a computed band gap of ≈ 2.06 eV for TiO2 (MP-390). This value is derived from density-functional-theory calculations and should be considered a theoretical estimate rather than an experimental measurement.
```

Preserve this working functionality. Do not rewrite it unnecessarily. Refactor only when doing so improves reliability, testability, or support for multiple tools.

## 10. Completed milestone: arXiv access

The `arxiv` Python package is installed and tested.

Tested code pattern:

```python
search = arxiv.Search(...)
client = arxiv.Client()
results = client.results(search)
```

Do not use the older deprecated `search.results()` pattern.

The current chosen literature topic is:

```text
High-entropy alloys (HEAs) — mechanical properties
```

The tested arXiv query is:

```python
'cat:cond-mat.mtrl-sci AND all:"high entropy alloy" AND all:mechanical'
```

This query returned relevant papers on strength and ductility, high-temperature mechanical behavior, yield strength, dislocation mechanisms, strain-rate effects, and hardness/ML-based HEA screening.

The project should initially use approximately 30–40 papers for a small, explainable portfolio-scale corpus.

## 11. RAG design decision

Version 1 should use **abstract-level RAG**, not full-PDF RAG.

Reason:

- simpler to implement
- easier to debug
- enough for a portfolio-scale corpus
- avoids introducing PDF extraction complexity too early

Each ingested paper should store at least:

```text
title
arXiv URL
abstract
published date
authors
arXiv ID
```

Useful optional metadata:

```text
updated date
categories
```

The first version should make it obvious that the retrieval corpus contains **abstracts**, not full paper text.

A later enhancement may add full-text PDFs.

## 12. Next phase: build the literature ingestion pipeline

Continue from the existing `ingest_arxiv.py`.

### Required behavior

1. Query arXiv using:

```python
QUERY = 'cat:cond-mat.mtrl-sci AND all:"high entropy alloy" AND all:mechanical'
```

2. Fetch about 40 papers.
3. Normalize each record into a plain Python dictionary.
4. Save a reproducible local dataset under `data/`, recommended as `data/hea_papers.json`.
5. Store title, abstract, URL, arXiv ID, authors, published date, and categories.
6. Avoid duplicate records by arXiv ID.
7. Print a concise ingestion summary.
8. Add error handling for network/API issues, zero search results, and malformed metadata.

### Acceptance test

Running:

```bash
python ingest_arxiv.py
```

should create a valid JSON file containing the HEA paper corpus.

## 13. Next phase: embeddings and vector database

After ingestion works, add the embedding/retrieval layer.

Preferred design:

```text
Voyage AI embeddings
ChromaDB
```

unless a current compatibility or pricing issue makes that unreasonable.

If Voyage is used, add `VOYAGE_API_KEY=...` to `.env`. Never hardcode it.

Required pipeline:

```text
paper abstract
    ↓
embedding model
    ↓
numeric vector
    ↓
ChromaDB
```

Store each abstract with metadata.

Persistent Chroma location:

```text
./chroma_db
```

This directory should remain Git-ignored.

## 14. Retrieval function

Create a clear retrieval function, likely in `rag.py` or `literature.py`.

Recommended interface:

```python
search_papers(query: str, k: int = 4)
```

It should:

1. embed the query
2. search ChromaDB
3. return the top-k results
4. include title, arXiv URL, relevant text/abstract excerpt, and retrieval score/distance if useful

Keep the return shape simple and JSON-serializable.

## 15. Add `search_papers` as a second LLM tool

The final agent should expose at least two tools:

```text
get_material
search_papers
```

Decision rule:

- use `get_material` for structured computed materials properties
- use `search_papers` for literature context, mechanisms, explanations, trends, and evidence

The model should be able to use one tool or both tools within one user query.

Do not force an unnatural combined demo merely to show both tools. The portfolio demo should be scientifically coherent.

## 16. Improve tool orchestration

The current code processes only the first tool call.

Refactor the tool loop so it can handle:

- zero tool calls
- one tool call
- multiple tool calls
- repeated tool-call rounds if the model requests another tool after receiving results
- unknown tool names
- malformed JSON tool arguments
- tool execution exceptions

Recommended pattern:

```text
while model requests tools:
    execute each requested tool
    append tool results
    call model again
return final response
```

Keep the implementation explicit and beginner-readable.

Avoid abstract agent frameworks for Version 1 unless they provide a clear benefit. Prefer direct SDK code so the owner can explain the control flow.

## 17. System prompt requirements

The assistant should be instructed to:

- use Materials Project for structured computed properties
- use literature retrieval for scientific context
- cite paper titles and URLs when literature is used
- distinguish computed values from experimental values
- avoid pretending one polymorph/property is universal
- state uncertainty when retrieval evidence is limited
- avoid inventing citations
- avoid claiming it read full papers when only abstracts are indexed

## 18. Testing requirements

Add automated tests before UI work.

Recommended test suite:

```text
tests/
├── test_materials_tool.py
├── test_arxiv_ingestion.py
├── test_retrieval.py
└── test_agent.py
```

Use `pytest`.

Tests should cover:

### Materials tool
- valid formula
- invalid formula
- multiple Materials Project structures
- stable-result selection behavior

### Ingestion
- expected fields exist
- duplicate arXiv IDs are removed
- JSON output is valid

### Retrieval
- query returns k or fewer results
- returned objects contain title/URL/excerpt
- obviously relevant HEA query retrieves HEA papers

### Agent
Use mocks where appropriate to avoid unnecessary API calls.

Test:
- no-tool response
- get_material tool call
- search_papers tool call
- multiple tool calls
- invalid tool arguments
- tool failure

## 19. Evaluation phase

Before building the UI, create a small manual evaluation set, recommended at `eval/questions.json`.

Include 10–20 representative questions across:

```text
property lookup
literature-only
mixed
ambiguous
invalid material
out-of-domain
```

Record expected tool(s), whether retrieval is required, whether the final answer is grounded, and whether citations are correct.

This evaluation artifact is important for AI-engineering interviews.

## 20. Streamlit phase

Only after the backend works reliably, build `app.py`.

Minimum UI:

- title
- short description
- chat history
- `st.chat_input`
- `st.chat_message`
- clear loading state
- graceful errors

Nice additions:

- expandable Sources section
- material ID links
- retrieved-paper links
- small badge showing which tool(s) were used
- reset conversation button

Do not overdesign the first UI.

## 21. Deployment

Target:

```text
Streamlit Community Cloud
```

Secrets must be configured through Streamlit secrets/environment configuration. Never commit `.env`.

Deployment should have:

- working public URL
- reproducible requirements
- no local-only path assumptions
- sensible error messages if an external API is unavailable

## 22. README requirements

The final `README.md` should include:

1. project title
2. one-paragraph problem statement
3. live demo link
4. screenshot/GIF
5. architecture diagram
6. why this project is different from generic RAG demos
7. feature list
8. tech stack
9. how tool calling works
10. how RAG works
11. data sources
12. setup instructions
13. environment variables
14. evaluation approach
15. limitations
16. future improvements

Important framing: this project combines a structured materials database with scientific-literature retrieval because materials engineers need both **the number** and **the scientific context behind the number**.

## 23. Suggested architecture

```text
                         USER
                          │
                          ▼
                    Groq-hosted LLM
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       get_material()          search_papers()
              │                       │
              ▼                       ▼
     Materials Project            ChromaDB
                                      │
                                      ▼
                             HEA arXiv abstracts
                                      ▲
                                      │
                              embedding pipeline
                                      ▲
                                      │
                                  arXiv API
```

## 24. Suggested final code structure

```text
materials-copilot/
├── app.py
├── copilot.py
├── tools.py
├── ingest_arxiv.py
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── hea_papers.json
├── eval/
│   └── questions.json
└── tests/
    ├── test_materials_tool.py
    ├── test_arxiv_ingestion.py
    ├── test_retrieval.py
    └── test_agent.py
```

Local-only:

```text
.env
.venv/
chroma_db/
```

## 25. Coding style constraints

Please follow these throughout the project:

- Python 3.12
- small functions
- descriptive names
- type hints where useful
- comments that explain WHY, not obvious syntax
- minimal global state
- no unnecessary frameworks
- no premature abstractions
- no giant files
- no hidden magic
- clear exception handling
- JSON-serializable tool outputs
- avoid hardcoded user paths
- do not silently swallow errors

Prefer code the owner can explain in an interview.

## 26. Git workflow

Work in small milestones.

Before each commit:

```bash
git status
```

Confirm `.env` is not staged.

Suggested commit sequence:

```text
Add HEA arXiv ingestion pipeline
Add embeddings and Chroma retrieval
Add literature search tool
Support multi-tool agent loop
Add automated tests
Add evaluation dataset
Add Streamlit chat interface
Add deployment configuration
Add project documentation
```

Do not squash everything into one giant commit during development.

## 27. What Codex should do first

Before changing code:

1. Inspect the entire existing repository.
2. Read this `PROJECT_HANDOFF.md`.
3. Read `tools.py`, `copilot.py`, `requirements.txt`, `test_arxiv.py`, and `ingest_arxiv.py`.
4. Run existing scripts only if safe.
5. Do not rewrite working Materials Project or Groq functionality without a reason.
6. Check `git status`.
7. Confirm `.env` is ignored.

Then implement **only the next milestone**:

```text
Complete the HEA arXiv ingestion pipeline and save a clean deduplicated JSON corpus under data/.
```

After that:

- run it
- inspect the output
- summarize changed files
- explain the code in plain language
- show `git diff`
- stop and wait for review before starting embeddings

## 28. Prompt to give Codex

Use this as the first task prompt:

```text
Read PROJECT_HANDOFF.md and inspect the existing repository before making changes.

Continue the Materials-Informatics Copilot from its current state.

For this task, implement ONLY the next milestone: complete the HEA arXiv ingestion pipeline.

Requirements:
- preserve the existing working Materials Project and Groq tool-calling code
- use the existing arXiv query from PROJECT_HANDOFF.md
- retrieve about 40 HEA mechanical-properties papers
- normalize metadata into JSON-serializable dictionaries
- deduplicate by arXiv ID
- save the corpus to data/hea_papers.json
- include title, abstract, arXiv URL, arXiv ID, authors, published date, and categories
- add clear error handling
- keep the code beginner-readable
- do not add embeddings, ChromaDB, Streamlit, or deployment yet
- run the ingestion script and validate the output
- show me which files changed and explain them
- do not commit automatically; stop for review when the milestone works
```

## 29. Definition of done for the overall project

The project is complete when:

- Materials Project lookup works
- arXiv corpus ingestion works
- embeddings are generated
- ChromaDB persists the literature corpus
- `search_papers()` retrieves relevant literature
- the LLM can call `get_material` and `search_papers`
- multi-tool queries work
- literature answers include source titles/URLs
- automated tests pass
- a small evaluation set exists
- Streamlit UI works locally
- app is deployed
- README is portfolio-ready
- secrets are not committed
- the owner can explain the architecture and key design decisions

## 30. Interview explanation target

By the end, the owner should be able to explain the project approximately like this:

> I built a materials-informatics copilot that combines structured property lookup from the Materials Project with retrieval over a curated corpus of high-entropy-alloy literature. The LLM uses tool calling to decide whether it needs a computed property, literature context, or both. For literature questions, the system embeds arXiv abstracts, stores them in a vector database, retrieves the most semantically relevant papers, and gives those results back to the model so the answer is grounded and cited. I kept the orchestration explicit rather than hiding it behind an agent framework so I could test and explain each step. I also added evaluation cases for tool selection, retrieval relevance, grounding, and failure handling.
