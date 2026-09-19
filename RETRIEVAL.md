# HEA abstract retrieval

This milestone adds standalone literature retrieval. It does not change the
Materials Project lookup or the Groq tool-calling code.

## Setup and use

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Add your real `VOYAGE_API_KEY` to the local `.env` file. Do not commit the key.
Then run:

```bash
.venv/bin/python rag.py index
.venv/bin/python rag.py search "What affects yield strength in high entropy alloys?" -k 4
.venv/bin/python -m pytest
```

Indexing reads `data/hea_papers.json`, sends abstracts to Voyage AI in batches of
10, and stores the vectors, abstracts, and metadata in `chroma_db/`. Both paths
are relative to the project directory, regardless of the terminal's directory.
The database and `.env` are Git-ignored.

For accounts with low rate limits, requests retry up to three times with a
30-second pause after each rate-limit response. Indexing can take a few minutes.

We use `voyage-4-lite` with `input_type="document"` for abstracts and
`input_type="query"` for searches, following the
[Voyage embedding documentation](https://docs.voyageai.com/docs/embeddings).
Chroma uses its [persistent local client](https://docs.trychroma.com/reference/python/client)
and [cosine distance](https://docs.trychroma.com/docs/collections/configure).

```python
from rag import search_papers

papers = search_papers("HEA strength and ductility", k=4)
```

Each result contains `arxiv_id`, `title`, `arxiv_url`, `authors`, `published`,
`excerpt`, and `distance`. The existing `url` field is retained as an alias for
`arxiv_url`. Citation metadata comes from the stored corpus, including authors
decoded from Chroma metadata.
The excerpt is the complete abstract, not full paper text. Lower distance means
a closer vector match; it is not a calibrated confidence or relevance score.
Results are limited to the number of indexed papers. Invalid queries, missing
keys, and missing or empty indexes produce clear errors.

Repeated indexing embeds the corpus again and updates records by arXiv ID,
without adding duplicates. Papers removed from the JSON are removed from the
index after successful upsert. All Voyage calls finish before database changes,
so an API failure preserves the previous index. Upsert and stale-record deletion
are separate database operations; rerun indexing after a database write failure.
Changing the embedding model requires a separate collection and reindexing.

The tests use actual temporary Chroma databases and mock Voyage responses. They
test storage, retrieval ranking with controlled vectors, metadata, input errors,
and API failure handling. They do not measure Voyage's semantic relevance.
The `pytest.ini` configuration excludes the root-level scripts that make live
Materials Project, Groq, and arXiv calls.

## Verified milestone

Validated with Voyage AI SDK 0.5.0, ChromaDB 1.5.9, and pytest 9.1.1:

- All 26 offline tests passed, including the existing ingestion tests.
- Live indexing saved 40 abstracts with 1,024-dimensional Voyage embeddings.
- A separate process reopened the database and checked all IDs, abstracts,
  titles, and URLs against the JSON corpus.
- A live query for "What affects yield strength in high entropy alloys?"
  returned four results. The first was "Yield strength insensitivity in a
  dual-phase high entropy alloy after prolonged high temperature annealing"
  (arXiv 2103.05567), with cosine distance approximately 0.428.

This is a live smoke check, not a comprehensive retrieval-quality evaluation.

## Literature grounding in the copilot

Literature answers use verified quotations rather than free-form scientific
synthesis. Groq selects an arXiv ID and exact passage from a retrieved abstract;
Python checks that the ID was retrieved and the passage occurs verbatim in that
abstract. Python then builds the citation from the stored title, URL, ID, and date.
Invalid passages, invented citation fields, and free-form responses are rejected
with an evidence-limit message. The full abstract remains available in `excerpt`.
This does not prove that a selected passage fully answers the question, but it
prevents the answer from adding unverified scientific wording. The response
explicitly limits its evidence to abstracts. After a literature tool response, the
next model request disables further tools and asks for final evidence selection;
this avoids repeated searches exhausting Groq’s token budget. Earlier material
tool rounds and multiple calls in one response remain supported. For mixed questions, computed
Materials Project values are shown separately, directly from the tool output.

Quotation matching normalizes whitespace, Unicode canonical equivalents, curly
quotes, and equivalent hyphens. It does not normalize scientific minus signs,
subscripts, numerical values, or substitute scientific terms. Every selected ID
must be retrieved and every normalized quotation must occur in that abstract.
An invalid or empty first selection receives one retry with validation feedback;
a second failed selection safely abstains. Evaluation records both reasons and
selector responses. Relevant quotations can support a cited, limited answer even
when requested details are absent, without inventing those details. Materials
Project results remain independently available in mixed answers.
