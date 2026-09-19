"""Run inspectable tool-selection and citation checks, without an LLM judge."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from copilot import MODEL, ask_material_question

ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RESULTS_PATH = ROOT / "eval" / "results.json"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation questions must be a non-empty list.")
    ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every evaluation case must be an object.")
        for key in ("id", "question", "category", "expected_grounding_behavior", "notes"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Missing or invalid case field: {key}")
        tools = case.get("expected_tools")
        if not isinstance(tools, list) or any(t not in ("get_material", "search_papers") for t in tools):
            raise ValueError("expected_tools must list known tools.")
        if type(case.get("literature_expected")) is not bool:
            raise ValueError("literature_expected must be a boolean.")
        if case["literature_expected"] != ("search_papers" in tools):
            raise ValueError("Literature expectation must agree with expected_tools.")
        if case["id"] in ids:
            raise ValueError("Evaluation case IDs must be unique.")
        ids.add(case["id"])
    return cases


def citation_matches(answer: str, trace: list[dict]) -> list[dict]:
    """Check exact title/URL citations against papers actually returned by a tool."""
    matches = {}
    for call in trace:
        if call["name"] != "search_papers" or not isinstance(call["result"], list):
            continue
        for paper in call["result"]:
            if not isinstance(paper, dict):
                continue
            title, url = paper.get("title"), paper.get("arxiv_url")
            if isinstance(title, str) and isinstance(url, str) and title and url:
                if f"[{title}]({url})" in answer:
                    matches[url] = {"title": title, "arxiv_url": url}
    return list(matches.values())


def evaluate_case(case: dict, agent=ask_material_question) -> dict:
    trace = []
    evidence_trace = []
    error = None
    answer = ""
    try:
        answer = agent(case["question"], tool_trace=trace, evidence_trace=evidence_trace)
        if not isinstance(answer, str):
            raise TypeError("Agent answer must be a string.")
    except Exception as exception:
        # Raw API exception messages can contain request details or credentials.
        error = type(exception).__name__
        answer = ""
    actual_tools = sorted({call["name"] for call in trace})
    citations = citation_matches(answer, trace)
    return {
        **case,
        "actual_tools": actual_tools,
        "tool_calls": trace,
        "evidence_selection": evidence_trace,
        "evidence_selection_outcome": evidence_trace[-1]["reason"] if evidence_trace else "not_attempted",
        "final_answer": answer,
        "tool_selection_matched": set(actual_tools) == set(case["expected_tools"]),
        "literature_citations_present": bool(citations),
        "citation_check_passed": bool(citations) if case["literature_expected"] else None,
        "matched_citations": citations,
        "execution_error": error,
        "manual_grounding_review": "not_scored",
    }


def save_results(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_evaluation(cases: list[dict], output: Path = RESULTS_PATH,
                   agent=ask_material_question, delay: float = 0) -> dict:
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "status": "running",
        "case_count": len(cases),
        "results": [],
    }
    for index, case in enumerate(cases):
        if index and delay:
            time.sleep(delay)
        result = evaluate_case(case, agent)
        report["results"].append(result)
        # Checkpoint after every question so later failures do not lose earlier work.
        save_results(report, output)
        print(f"{case['id']}: tools={result['actual_tools']}, matched={result['tool_selection_matched']}, error={result['execution_error']}", flush=True)
    report["status"] = "completed"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_results(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--delay", type=float, default=25, help="Seconds between live cases (default 25).")
    args = parser.parse_args()
    if not 0 <= args.delay <= 60:
        parser.error("--delay must be between 0 and 60 seconds")
    run_evaluation(load_questions(args.questions), args.output, delay=args.delay)
    print(f"Saved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
