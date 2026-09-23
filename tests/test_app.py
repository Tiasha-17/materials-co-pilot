"""Exercise the Streamlit interface without spending API credits."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import copilot
from app import get_reply

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_safe_reply_metadata(monkeypatch):
    def agent(question, *, tool_trace):
        tool_trace.extend([
            {"name": "search_papers", "result": "hidden diagnostics"},
            {"name": "search_papers"}, {"name": "get_material"},
            {"name": "untrusted tool label"},
        ])
        return "[Paper](https://arxiv.org/abs/2401.00001)"
    monkeypatch.setattr(copilot, "ask_material_question", agent)
    reply = get_reply("HEA?")
    assert reply["tools"] == ["Literature search", "Materials Project"]
    assert "hidden diagnostics" not in str(reply)
    assert "untrusted" not in str(reply)
    assert reply["content"] == "[Paper](https://arxiv.org/abs/2401.00001)"


def test_chat_history_clear_and_no_duplicate_calls(monkeypatch):
    backend = Mock(return_value="[Evidence](https://arxiv.org/abs/2401.00001)")
    monkeypatch.setattr(copilot, "ask_material_question", backend)
    at = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not at.exception
    assert at.title[0].value == "Materials-Informatics Copilot"
    at.chat_input[0].set_value("HEA strength?").run()
    assert not at.exception
    assert len(at.chat_message) == 2
    assert "https://arxiv.org/abs/2401.00001" in at.chat_message[1].markdown[0].value
    at.run()
    assert len(at.chat_message) == 2
    backend.assert_called_once()
    at.chat_input[0].set_value("Another question").run()
    assert len(at.chat_message) == 4
    at.button[0].click().run()
    assert len(at.chat_message) == 0
    assert at.session_state["messages"] == []
    assert backend.call_count == 2


def test_error_is_safe_and_app_recovers(monkeypatch):
    backend = Mock(side_effect=[RuntimeError("SECRET credential debug information"), "Recovered answer"])
    monkeypatch.setattr(copilot, "ask_material_question", backend)
    at = AppTest.from_file(str(APP), default_timeout=20).run()
    at.chat_input[0].set_value("First question").run()
    assert not at.exception
    assert len(at.error) == 1
    assert "SECRET" not in at.error[0].value
    assert "SECRET" not in str(at.session_state["messages"])
    at.chat_input[0].set_value("Try again").run()
    assert not at.exception
    assert at.chat_message[-1].markdown[0].value == "Recovered answer"


@pytest.mark.parametrize("present", [True, False])
def test_diagnostics_log_only_exception_type(monkeypatch, caplog, present):
    keys = ("MP_API_KEY", "GROQ_API_KEY", "VOYAGE_API_KEY")
    secrets = ["private-mp-value", "private-groq-value", "private-voyage-value"]
    for key, value in zip(keys, secrets):
        if present:
            monkeypatch.setenv(key, value)
        else:
            monkeypatch.delenv(key, raising=False)
    error = RuntimeError("Authorization: Bearer " + " ".join(secrets)
                         + " https://user:password@example.com request-body")
    monkeypatch.setattr(copilot, "ask_material_question", Mock(side_effect=error))

    reply = get_reply("private user question")

    assert reply["error"] is True
    assert reply["content"] == (
        "I couldn't complete this request. Please try again shortly. "
        "If the issue persists, check your local API configuration and literature index."
    )
    records = [record for record in caplog.records if record.name == "app"]
    assert len(records) == 1
    record = records[0]
    assert record.getMessage() == (
        "exception_module=builtins exception_class=RuntimeError"
    )
    assert record.exc_info is None
    assert record.exc_text is None
    assert record.stack_info is None
    assert all(isinstance(value, (str, bool)) for value in record.args)
    for forbidden in list(keys) + secrets + ["Authorization", "password", "request-body", "private user question"]:
        assert forbidden not in caplog.text
        assert forbidden not in str(record.args)
        assert forbidden not in str(reply)
