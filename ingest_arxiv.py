"""Download HEA paper metadata and abstracts (not full paper text)."""

import json
from pathlib import Path
import re
import sys

import arxiv
from requests.exceptions import RequestException


QUERY = 'cat:cond-mat.mtrl-sci AND all:"high entropy alloy" AND all:mechanical'
MAX_PAPERS = 40
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "hea_papers.json"


def clean_text(value: str) -> str:
    """Remove API line wrapping and reject missing text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Required text is missing or empty")
    return " ".join(value.split())


def normalize_paper(result: arxiv.Result) -> dict:
    """Convert SDK objects into plain, JSON-serializable metadata."""
    short_id = result.get_short_id()
    # Versions v1 and v2 belong to the same paper, including legacy IDs.
    arxiv_id = re.sub(r"v\d+$", "", short_id)
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})", arxiv_id):
        raise ValueError(f"Invalid arXiv ID: {short_id!r}")
    if not result.authors or not result.categories:
        raise ValueError("Authors or categories are missing")

    return {
        "title": clean_text(result.title),
        "abstract": clean_text(result.summary),
        "arxiv_url": f"https://arxiv.org/abs/{short_id}",
        "arxiv_id": arxiv_id,
        "authors": [clean_text(author.name) for author in result.authors],
        "published": result.published.date().isoformat(),
        "categories": [clean_text(category) for category in result.categories],
    }


def fetch_papers() -> list[dict]:
    """Collect up to 40 unique, valid papers in relevance order."""
    search = arxiv.Search(query=QUERY, sort_by=arxiv.SortCriterion.Relevance)
    client = arxiv.Client(page_size=MAX_PAPERS, delay_seconds=3, num_retries=3)
    papers = []
    seen_ids = set()
    duplicates = 0
    malformed = 0

    try:
        # Continue past duplicates or invalid records to fill the target count.
        for result in client.results(search):
            try:
                paper = normalize_paper(result)
            except (AttributeError, TypeError, ValueError) as error:
                malformed += 1
                print(f"Warning: skipping malformed paper: {error}", file=sys.stderr)
                continue
            if paper["arxiv_id"] in seen_ids:
                duplicates += 1
                continue
            seen_ids.add(paper["arxiv_id"])
            papers.append(paper)
            if len(papers) == MAX_PAPERS:
                break
    except (arxiv.ArxivError, RequestException) as error:
        raise RuntimeError(f"arXiv retrieval failed; retry later. Details: {error}") from error

    if not papers:
        raise RuntimeError("No valid papers returned. Check the query and metadata warnings.")
    print(f"Retrieved {len(papers)} papers; skipped {duplicates} duplicates and {malformed} malformed records.")
    if len(papers) < MAX_PAPERS:
        print(f"Warning: search exhausted before reaching {MAX_PAPERS} valid papers.", file=sys.stderr)
    return papers


def save_papers(papers: list[dict], output_path: Path = OUTPUT_PATH) -> None:
    """Replace the corpus only after the complete JSON file has been written."""
    payload = json.dumps(papers, ensure_ascii=False, indent=2) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    try:
        papers = fetch_papers()
        save_papers(papers)
    except (RuntimeError, OSError, TypeError, ValueError) as error:
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved abstracts and metadata to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
