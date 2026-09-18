"""Use real local Chroma storage and deterministic, offline Voyage responses."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import rag


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    papers = [
        {
            "arxiv_id": "2401.00001", "title": "HEA yield strength",
            "abstract": "Yield strength in high entropy alloys.",
            "arxiv_url": "https://arxiv.org/abs/2401.00001",
            "authors": ["A. Author"], "published": "2024-01-01",
            "categories": ["cond-mat.mtrl-sci"],
        },
        {
            "arxiv_id": "2401.00002", "title": "HEA thermal transport",
            "abstract": "Thermal conductivity in high entropy alloys.",
            "arxiv_url": "https://arxiv.org/abs/2401.00002",
            "authors": ["B. Author"], "published": "2024-01-02",
            "categories": ["cond-mat.mtrl-sci"],
        },
    ]
    path = tmp_path / "papers.json"
    path.write_text(json.dumps(papers))
    monkeypatch.setattr(rag, "CORPUS_PATH", path)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "chroma_db")
    monkeypatch.setattr(rag, "load_dotenv", lambda *args: None)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-placeholder")

    def embed(texts, **kwargs):
        vectors = [[1.0, 0.0] if "strength" in text.lower() else [0.0, 1.0]
                   for text in texts]
        return SimpleNamespace(embeddings=vectors)

    client = Mock()
    client.embed.side_effect = embed
    monkeypatch.setattr(rag.voyageai, "Client", Mock(return_value=client))
    return papers, client


def test_index_persistence_and_repeated_indexing(corpus):
    papers, client = corpus
    assert rag.index_papers() == 2
    assert rag.index_papers() == 2
    assert (rag.DB_PATH / "chroma.sqlite3").exists()
    # A newly opened client can read the persisted data.
    collection = rag.get_collection()
    stored = collection.get(ids=[papers[0]["arxiv_id"]], include=["documents", "metadatas"])
    assert stored["documents"] == [papers[0]["abstract"]]
    assert json.loads(stored["metadatas"][0]["authors"]) == papers[0]["authors"]
    assert client.embed.call_args.kwargs["input_type"] == "document"
    assert client.embed.call_args.kwargs["model"] == rag.MODEL


def test_retrieval_ranking_shape_and_limit(corpus):
    _, client = corpus
    rag.index_papers()
    results = rag.search_papers("HEA yield strength", k=1)
    assert len(results) == 1
    assert results[0]["title"] == "HEA yield strength"
    assert results[0]["distance"] == pytest.approx(0, abs=1e-6)
    assert set(results[0]) == {"arxiv_id", "title", "url", "arxiv_url", "authors", "published", "excerpt", "distance"}
    assert results[0]["excerpt"] == corpus[0][0]["abstract"]
    json.dumps(results, allow_nan=False)
    assert client.embed.call_args.kwargs["input_type"] == "query"
    assert len(rag.search_papers("HEA yield strength", k=10)) == 2


def test_reindex_updates_and_removes_old_papers(corpus):
    papers, _ = corpus
    rag.index_papers()
    papers[0]["title"] = "Updated title"
    rag.CORPUS_PATH.write_text(json.dumps(papers[:1]))
    assert rag.index_papers() == 1
    assert rag.search_papers("strength")[0]["title"] == "Updated title"


def test_api_failure_preserves_index(corpus):
    _, client = corpus
    rag.index_papers()
    client.embed.side_effect = RuntimeError("Mock API failure")
    with pytest.raises(RuntimeError, match="Voyage embedding request failed"):
        rag.index_papers()
    assert rag.get_collection().count() == 2


def test_missing_credentials(corpus, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY")
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        rag.index_papers()
    assert not rag.DB_PATH.exists()


def test_missing_and_empty_index(corpus):
    with pytest.raises(RuntimeError, match="No local index"):
        rag.search_papers("strength")
    rag.get_collection(create=True)
    with pytest.raises(RuntimeError, match="empty"):
        rag.search_papers("strength")


@pytest.mark.parametrize("query,k", [("", 4), (" ", 4), (None, 4), ("HEA", 0),
                                      ("HEA", -1), ("HEA", True), ("HEA", 1.5)])
def test_invalid_search_inputs(query, k):
    with pytest.raises(ValueError):
        rag.search_papers(query, k)


@pytest.mark.parametrize("contents", ["bad JSON", "[]", "{}", '[{"title": "incomplete"}]'])
def test_invalid_corpus(corpus, contents):
    _, client = corpus
    rag.CORPUS_PATH.write_text(contents)
    with pytest.raises(ValueError):
        rag.index_papers()
    client.embed.assert_not_called()


def test_duplicate_corpus_ids(corpus):
    papers, _ = corpus
    rag.CORPUS_PATH.write_text(json.dumps([papers[0], papers[0]]))
    with pytest.raises(ValueError, match="Duplicate"):
        rag.index_papers()


def test_model_mismatch(corpus):
    rag.index_papers()
    rag.get_collection().modify(metadata={"embedding_model": "different-model"})
    with pytest.raises(RuntimeError, match="does not match"):
        rag.search_papers("strength")


def test_rate_limit_retry(corpus, monkeypatch):
    _, client = corpus
    sleep = Mock()
    monkeypatch.setattr(rag.time, "sleep", sleep)
    client.embed.side_effect = [
        rag.RateLimitError("Rate limited"),
        SimpleNamespace(embeddings=[[1.0, 0.0]]),
    ]
    assert rag.embed_texts(["strength"], "query") == [[1.0, 0.0]]
    sleep.assert_called_once_with(30)


def test_rate_limit_retries_are_bounded(corpus, monkeypatch):
    _, client = corpus
    sleep = Mock()
    monkeypatch.setattr(rag.time, "sleep", sleep)
    client.embed.side_effect = rag.RateLimitError("Rate limited")
    with pytest.raises(RuntimeError, match="rate limit persisted"):
        rag.index_papers()
    assert client.embed.call_count == 4
    assert sleep.call_count == 3
    assert not rag.DB_PATH.exists()


def test_citation_metadata_matches_corpus(corpus):
    papers, _ = corpus
    rag.index_papers()
    source = {paper["arxiv_id"]: paper for paper in papers}
    for result in rag.search_papers("strength"):
        paper = source[result["arxiv_id"]]
        for field in ("title", "arxiv_url", "arxiv_id", "authors", "published"):
            assert result[field] == paper[field]
        assert result["url"] == result["arxiv_url"]
        assert result["excerpt"] == paper["abstract"]
        json.dumps(result, allow_nan=False)
