# Small copilot evaluation

Run the 16 live cases through the existing agent:

```bash
.venv/bin/python evaluate.py
```

This uses the configured Groq, Materials Project, and Voyage credentials and the
existing Chroma index. It makes real API requests. The default 25-second interval
reduces rate-limit pressure; provider failures are recorded and later cases continue.
`--questions`, `--output`, and `--delay` override the defaults. Results are saved to
`eval/results.json` after every case, then marked completed when all cases have
been attempted. Running again replaces this local report.

Cases cover property lookup, literature, mixed requests, ambiguity, invalid
materials, and out-of-domain questions. Expected tool choices are reviewable
policies, not universal truths: an invalid-material question may reasonably be
rejected without a lookup, for example. Each case includes notes and expected
scientific grounding for human review.

The report contains each final answer, a tool trace, and simple checks:

- `actual_tools` is the unique set of tool names dispatched/requested by the
  agent loop, including calls that return errors. It does not imply success.
  Each trace entry records `status` and the returned result. Skipped calls at the
  per-round limit appear as error results too.
- `tool_selection_matched` compares exact sets, ignoring order and duplicate
  calls. It does not check formula/query correctness or required call counts.
- `literature_citations_present` checks for an exact Markdown title/URL citation
  matching a paper returned by `search_papers` in that case.
- `citation_check_passed` is that check for literature-expected cases, otherwise
  null. A justified abstention can fail this check; review it manually.
- `execution_error` records only the exception class, not raw API error text.
  Tool-level failures can instead appear in the trace without an agent exception.

There is no overall quality score or LLM judge. Citation presence is not proof
that all claims are correct, all citations are valid, or the answer is complete.
Review `expected_grounding_behavior` and `notes` against the answer and returned
abstracts. A completed report means every case was attempted, not that every case
passed. Normal unit tests mock the agent/APIs and write only to temporary folders.

Evidence selection diagnostics are saved separately from tool calls in
`evidence_selection`. Each attempt records the selector response and one reason:
`no_evidence_selected`, `malformed_evidence_json`, `unknown_arxiv_id`,
`quotation_mismatch`, or `successful_grounded_evidence`. A retry API failure is
reported as `evidence_api_error` with only its exception class.
`evidence_selection_outcome` identifies the last recorded reason or `not_attempted`.
These diagnostics are for review, not normal user answers. There is at most one
selection retry, with explicit validation feedback. Cases 13 and 14 now expect
no tool calls for obvious invalid formulas; no database absence claim is warranted
without a lookup.
