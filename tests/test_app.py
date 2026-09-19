"""Exercise the Streamlit interface without spending API credits."""

from pathlib import Path
from unittest.mock import Mock

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
