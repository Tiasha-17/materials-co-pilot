"""Embed HEA abstracts with Voyage and search a persistent local Chroma index."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import chromadb
from chromadb.errors import NotFoundError
from dotenv import load_dotenv
import voyageai
from voyageai.error import RateLimitError


PROJECT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = PROJECT_DIR / "data" / "hea_papers.json"
DB_PATH = PROJECT_DIR / "chroma_db"
MODEL = "voyage-4-lite"
COLLECTION_NAME = "hea_abstracts_voyage_4_lite"
BATCH_SIZE = 10


def load_papers(path: Path) -> list[dict]:
    """Validate the entire corpus before making any paid API requests."""
    try:
        papers = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read corpus at {path}: {error}") from error
    if not isinstance(papers, list) or not papers:
        raise ValueError("Corpus must be a non-empty JSON list.")
    seen = set()
    for number, paper in enumerate(papers, start=1):
        if not isinstance(paper, dict):
            raise ValueError(f"Paper {number} must be a JSON object.")
        for field in ("arxiv_id", "title", "abstract", "arxiv_url", "published"):
            if not isinstance(paper.get(field), str) or not paper[field].strip():
                raise ValueError(f"Paper {number} has missing or invalid {field}.")
        for field in ("authors", "categories"):
            values = paper.get(field)
            if not isinstance(values, list) or not values or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise ValueError(f"Paper {number} has missing or invalid {field}.")
        if paper["arxiv_id"] in seen:
            raise ValueError(f"Duplicate arXiv ID: {paper['arxiv_id']}")
        seen.add(paper["arxiv_id"])
    return papers


def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Use document embeddings for abstracts and query embeddings for searches."""
    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("VOYAGE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set VOYAGE_API_KEY in your local .env before indexing or searching.")
    try:
        client = voyageai.Client(api_key=api_key, max_retries=0, timeout=60)
        for attempt in range(4):
            try:
                result = client.embed(
                    texts, model=MODEL, input_type=input_type, truncation=False
                )
                break
            except RateLimitError:
                if attempt == 3:
                    raise
                print("Voyage rate limit reached; retrying in 30 seconds.", file=sys.stderr)
                time.sleep(30)
    except RateLimitError:
        raise RuntimeError(
            "Voyage rate limit persisted after retries. Wait a minute "
            "or check your account's token/request limits."
        ) from None
    except Exception as error:
        # Avoid including HTTP headers or credential-bearing SDK details in output.
        raise RuntimeError(
            "Voyage embedding request failed. Check your key, network, quota, "
            "and model access, then retry."
        ) from error
    if len(result.embeddings) != len(texts):
        raise RuntimeError("Voyage returned an unexpected number of embeddings.")
    return result.embeddings


def get_collection(create: bool = False):
    """Disable Chroma's default embedding model: vectors always come from Voyage."""
    if not create and not DB_PATH.exists():
        raise RuntimeError("No local index found. Run: python rag.py index")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    if create:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"embedding_model": MODEL, "content": "abstract"},
            configuration={"hnsw": {"space": "cosine"}},
        )
    else:
        try:
            collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        except NotFoundError as error:
            raise RuntimeError("No HEA index found. Run: python rag.py index") from error
    if (collection.metadata or {}).get("embedding_model") != MODEL:
        raise RuntimeError("Index embedding model does not match the configured model.")
    return collection


def index_papers() -> int:
    """Upsert by arXiv ID, then remove papers no longer present in the corpus."""
    papers = load_papers(CORPUS_PATH)
    embeddings = []
    # Finish all API calls before changing the stored corpus.
    for start in range(0, len(papers), BATCH_SIZE):
        batch = papers[start:start + BATCH_SIZE]
        embeddings.extend(embed_texts([paper["abstract"] for paper in batch], "document"))

    collection = get_collection(create=True)
    ids = [paper["arxiv_id"] for paper in papers]
    metadata = []
    for paper in papers:
        metadata.append({
            "title": paper["title"],
            "url": paper["arxiv_url"],
            "published": paper["published"],
            # JSON strings work with Chroma versions that require scalar metadata.
            "authors": json.dumps(paper["authors"], ensure_ascii=False),
            "categories": json.dumps(paper["categories"]),
        })
    collection.upsert(
        ids=ids, embeddings=embeddings,
        documents=[paper["abstract"] for paper in papers], metadatas=metadata,
    )
    stale_ids = sorted(set(collection.get(include=[])["ids"]) - set(ids))
    if stale_ids:
        collection.delete(ids=stale_ids)
    return collection.count()


def search_papers(query: str, k: int = 4) -> list[dict]:
    """Return up to k abstracts; lower cosine distance means a closer match."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    collection = get_collection()
    count = collection.count()
    if count == 0:
        raise RuntimeError("The HEA index is empty. Run: python rag.py index")
    result = collection.query(
        query_embeddings=embed_texts([query.strip()], "query"),
        n_results=min(k, count), include=["documents", "metadatas", "distances"],
    )
    papers = []
    for index, arxiv_id in enumerate(result["ids"][0]):
        metadata = result["metadatas"][0][index]
        papers.append({
            "arxiv_id": arxiv_id,
            "title": metadata["title"],
            "url": metadata["url"],
            "arxiv_url": metadata["url"],
            "authors": json.loads(metadata["authors"]),
            "published": metadata["published"],
            # The full abstract avoids cutting off the evidence mid-sentence.
            "excerpt": result["documents"][0][index],
            "distance": float(result["distances"][0][index]),
        })
    return papers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Embed and store the existing JSON corpus")
    search = commands.add_parser("search", help="Search indexed abstracts")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.command == "index":
            print(f"Indexed {index_papers()} abstracts in {DB_PATH}")
        else:
            print(json.dumps(search_papers(args.query, args.k), indent=2, ensure_ascii=False))
    except Exception as error:
        print(f"Retrieval command failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
