"""Offline checks for evaluation utilities and the agent's optional tool trace."""

import json
from unittest.mock import Mock

from groq.types.chat import ChatCompletionMessage
import pytest

import copilot
import evaluate


@pytest.fixture
def case():
    return evaluate.load_questions()[3]


def fake_agent(question, *, tool_trace, evidence_trace=None):
    paper = {"title": "Real title", "arxiv_url": "https://arxiv.org/abs/2401.00001"}
    tool_trace.append({"name": "search_papers", "call_id": "a", "status": "ok", "result": [paper]})
    return "Evidence [Real title](https://arxiv.org/abs/2401.00001)"


def test_question_coverage():
    cases = evaluate.load_questions()
    assert 12 <= len(cases) <= 20
    assert {case["category"] for case in cases} == {
        "property_lookup", "literature_only", "mixed", "ambiguous", "invalid_material", "out_of_domain",
    }


def test_matching_tools_and_real_citations(case):
    result = evaluate.evaluate_case(case, fake_agent)
    assert result["tool_selection_matched"] is True
    assert result["citation_check_passed"] is True
    assert result["manual_grounding_review"] == "not_scored"
    assert result["execution_error"] is None


def test_unretrieved_or_fake_citation_does_not_pass(case):
    def agent(question, *, tool_trace, evidence_trace=None):
        fake_agent(question, tool_trace=tool_trace)
        return "[Fake title](https://arxiv.org/abs/2401.00001)"
    assert evaluate.evaluate_case(case, agent)["citation_check_passed"] is False
    assert evaluate.citation_matches("[Real title](https://arxiv.org/abs/2401.00001)", []) == []


def test_missing_tools_and_citations(case):
    result = evaluate.evaluate_case(case, lambda question, **kwargs: "Insufficient evidence.")
    assert result["actual_tools"] == []
    assert result["tool_selection_matched"] is False
    assert result["citation_check_passed"] is False


def test_no_tool_case_does_not_require_citation():
    case = evaluate.load_questions()[-1]
    result = evaluate.evaluate_case(case, lambda question, **kwargs: "102")
    assert result["tool_selection_matched"] is True
    assert result["citation_check_passed"] is None


def test_failed_tool_is_visible_not_success(case):
    def agent(question, *, tool_trace, evidence_trace=None):
        tool_trace.append({"name": "search_papers", "call_id": "a", "status": "error", "result": {"error": "unavailable"}})
        raise RuntimeError("sensitive request details")
    result = evaluate.evaluate_case(case, agent)
    assert result["tool_selection_matched"] is True
    assert result["tool_calls"][0]["status"] == "error"
    assert result["execution_error"] == "RuntimeError"
    assert "sensitive" not in json.dumps(result)


def test_repeated_calls_compared_as_sets(case):
    def agent(question, *, tool_trace, evidence_trace=None):
        fake_agent(question, tool_trace=tool_trace)
        return fake_agent(question, tool_trace=tool_trace)
    result = evaluate.evaluate_case(case, agent)
    assert result["tool_selection_matched"] is True
    assert len(result["tool_calls"]) == 2


def test_extra_tool_is_mismatch(case):
    def agent(question, *, tool_trace, evidence_trace=None):
        tool_trace.append({"name": "get_material", "status": "ok", "result": {}})
        return fake_agent(question, tool_trace=tool_trace)
    assert evaluate.evaluate_case(case, agent)["tool_selection_matched"] is False


def test_runner_saves_and_continues_after_error(case, tmp_path):
    second = {**case, "id": "second"}
    calls = 0
    def agent(question, *, tool_trace, evidence_trace=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("failure")
        return fake_agent(question, tool_trace=tool_trace)
    path = tmp_path / "results.json"
    report = evaluate.run_evaluation([case, second], path, agent=agent)
    assert json.loads(path.read_text()) == report
    assert report["status"] == "completed"
    assert report["results"][0]["execution_error"] == "RuntimeError"
    assert report["results"][1]["execution_error"] is None


@pytest.mark.parametrize("contents", [[], {}, [{"id": "missing_fields"}]])
def test_invalid_questions(contents, tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError):
        evaluate.load_questions(path)


def test_duplicate_ids_rejected(case, tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([case, case]))
    with pytest.raises(ValueError, match="unique"):
        evaluate.load_questions(path)


def test_agent_trace_does_not_change_answer(monkeypatch):
    monkeypatch.setattr(copilot, "get_material", Mock(return_value={"band_gap": 2.06}))
    tool_message = ChatCompletionMessage(role="assistant", tool_calls=[{
        "id": "a", "type": "function",
        "function": {"name": "get_material", "arguments": '{"formula":"TiO2"}'},
    }])
    final = ChatCompletionMessage(role="assistant", content="Computed value: 2.06 eV")
    client = Mock()
    client.chat.completions.create.side_effect = [Mock(choices=[Mock(message=m)]) for m in (tool_message, final)]
    trace = []
    assert copilot.ask_material_question("TiO2?", client=client, tool_trace=trace) == final.content
    assert trace == [{"name": "get_material", "call_id": "a", "status": "ok", "result": {"band_gap": 2.06}}]


def test_evidence_diagnostics_saved(case):
    def agent(question, *, tool_trace, evidence_trace):
        evidence_trace.extend([
            {'attempt': 1, 'reason': 'quotation_mismatch', 'response': 'bad selection'},
            {'attempt': 2, 'reason': 'successful_grounded_evidence', 'response': 'valid selection'},
        ])
        return fake_agent(question, tool_trace=tool_trace)
    result = evaluate.evaluate_case(case, agent)
    assert result['evidence_selection_outcome'] == 'successful_grounded_evidence'
    assert result['evidence_selection'][0]['reason'] == 'quotation_mismatch'
    assert result['citation_check_passed'] is True


def test_invalid_material_cases_expect_no_tools():
    cases = evaluate.load_questions()
    for case in cases:
        if case['id'] in ('case_13', 'case_14'):
            assert case['expected_tools'] == []
            assert 'unless an actual lookup was performed' in case['expected_grounding_behavior']
