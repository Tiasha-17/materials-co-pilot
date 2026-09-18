"""Offline ingestion checks: python -m unittest discover -s tests."""

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import arxiv
from requests.exceptions import ConnectionError

import ingest_arxiv as ingestion


def paper(version="v1", **changes):
    fields = {
        "entry_id": f"https://arxiv.org/abs/2401.12345{version}",
        "title": "A title\nwith wrapping",
        "summary": "An abstract about mechanical properties.",
        "authors": [arxiv.Result.Author("Example Author")],
        "published": datetime(2024, 1, 23, tzinfo=timezone.utc),
        "categories": ["cond-mat.mtrl-sci"],
    }
    fields.update(changes)
    return arxiv.Result(**fields)


class IngestionTests(unittest.TestCase):
    def test_metadata_and_json_round_trip(self):
        record = ingestion.normalize_paper(paper())
        self.assertEqual(record["arxiv_id"], "2401.12345")
        self.assertEqual(record["published"], "2024-01-23")
        self.assertEqual(record["title"], "A title with wrapping")
        self.assertEqual(record["authors"], ["Example Author"])
        self.assertEqual(set(record), {
            "title", "abstract", "arxiv_url", "arxiv_id",
            "authors", "published", "categories",
        })
        with TemporaryDirectory() as folder:
            path = Path(folder) / "data" / "hea_papers.json"
            ingestion.save_papers([record], path)
            self.assertEqual(json.loads(path.read_text()), [record])

    @patch("ingest_arxiv.arxiv.Client")
    def test_duplicates_and_malformed_metadata(self, client):
        client.return_value.results.return_value = iter([
            paper(summary=" "), paper("v1"), paper("v2"),
        ])
        self.assertEqual(len(ingestion.fetch_papers()), 1)

    @patch("ingest_arxiv.arxiv.Client")
    def test_empty_search(self, client):
        client.return_value.results.return_value = iter([])
        with self.assertRaisesRegex(RuntimeError, "No valid papers"):
            ingestion.fetch_papers()

    @patch("ingest_arxiv.save_papers")
    @patch("ingest_arxiv.arxiv.Client")
    def test_failure_mid_retrieval_does_not_save(self, client, save):
        def interrupted_results():
            yield paper()
            raise ConnectionError("Connection lost")

        client.return_value.results.return_value = interrupted_results()
        self.assertEqual(ingestion.main(), 1)
        save.assert_not_called()

    def test_failed_write_preserves_existing_corpus(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "hea_papers.json"
            path.write_text("[]\n")
            with patch.object(Path, "replace", side_effect=OSError("Disk error")):
                with self.assertRaises(OSError):
                    ingestion.save_papers([ingestion.normalize_paper(paper())], path)
            self.assertEqual(path.read_text(), "[]\n")
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
