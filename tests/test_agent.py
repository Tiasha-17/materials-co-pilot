"""Mock Groq and both tools; these tests never call external services."""

from copy import deepcopy
import json
from unittest.mock import Mock

from groq.types.chat import ChatCompletionMessage
import pytest

import copilot


def call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def message(calls=None, content=None):
    return ChatCompletionMessage(role="assistant", content=content, tool_calls=calls)


def client_for(*messages):
    client = Mock()
    history = []
    responses = iter(messages)

    def create(**kwargs):
        history.append(deepcopy(kwargs))
        return Mock(choices=[Mock(message=next(responses))])

    client.chat.completions.create.side_effect = create
    return client, history


def tool_results(request):
    """Read results from either the agent transcript or final extraction input."""
    if "response_format" in request:
        return json.loads(request["messages"][1]["content"])["tool_results"]
    return [m for m in request["messages"] if m["role"] == "tool"]


@pytest.fixture(autouse=True)
def mocked_tools(monkeypatch):
    material = Mock(return_value={"material_id": "mp-390", "band_gap": 2.06})
    literature = Mock(return_value=[{"title": "HEA study", "url": "https://arxiv.org/abs/2401.00001",
                                    "excerpt": "An abstract.", "distance": 0.2}])
    monkeypatch.setattr(copilot, "get_material", material)
    monkeypatch.setattr(copilot, "search_papers", literature)
    # Fail immediately if a test forgets to inject a Groq client.
    monkeypatch.setattr(copilot, "Groq", Mock(side_effect=AssertionError("Live client forbidden")))
    return material, literature


def test_no_tool_response(mocked_tools):
    client, history = client_for(message(content="Hello"))
    assert copilot.ask_material_question("Hello", client=client) == "Hello"
    assert len(history) == 1
    for tool in mocked_tools:
        tool.assert_not_called()
    assert {t["function"]["name"] for t in history[0]["tools"]} == {"get_material", "search_papers"}


@pytest.mark.parametrize("name,args", [
    ("get_material", '{"formula":"TiO2"}'),
    ("search_papers", '{"query":"HEA strength"}'),
    ("search_papers", '{"query":"HEA strength","k":2}'),
])
def test_single_tool(name, args, mocked_tools):
    client, history = client_for(message([call(name, args)]), message(content="Answer"))
    answer = copilot.ask_material_question("Question", client=client)
    assert answer == "Answer" if name == "get_material" else "verified evidence" in answer
    result = tool_results(history[1])[-1]
    assert result["tool_call_id"] == "call_1"
    selected = mocked_tools[0 if name == "get_material" else 1]
    expected = json.loads(args)
    if name == "search_papers":
        expected.setdefault("k", 4)
    selected.assert_called_once_with(**expected)
    assert json.loads(result["content"]) == selected.return_value


def test_multiple_calls_and_repeated_rounds(mocked_tools):
    client, history = client_for(
        message([call("get_material", '{"formula":"TiO2"}', "a"),
                 call("search_papers", '{"query":"HEA strength"}', "b")]),
        message([call("search_papers", '{"query":"HEA ductility"}', "c")]),
        message(content="Final answer"),
    )
    assert "verified evidence" in copilot.ask_material_question("Question", client=client)
    assert [m["tool_call_id"] for m in tool_results(history[1])] == ["a", "b"]
    assert [m["tool_call_id"] for m in tool_results(history[2])] == ["a", "b", "c"]
    assert mocked_tools[1].call_count == 2
    assert history[0]["tool_choice"] == "auto"
    assert all(h["tool_choice"] == "none" for h in history[1:])


@pytest.mark.parametrize("name,args", [
    ("unknown", "{}"), ("get_material", "bad JSON"),
    ("get_material", "[]"), ("get_material", "null"),
    ("get_material", "{}"), ("get_material", '{"formula":1}'),
    ("get_material", '{"formula":" "}'),
    ("get_material", '{"formula":"TiO2","extra":true}'),
    ("search_papers", '{"query":"HEA","k":true}'),
    ("search_papers", '{"query":"HEA","k":0}'),
    ("search_papers", '{"query":"HEA","k":11}'),
    ("search_papers", '{"query":"HEA","k":1.5}'),
])
def test_bad_calls_return_errors_and_continue(name, args, mocked_tools):
    client, history = client_for(message([call(name, args)]), message(content="Recovered"))
    answer = copilot.ask_material_question("Question", client=client)
    assert answer == "Recovered" if name != "search_papers" else "verified evidence" in answer
    assert "error" in json.loads(tool_results(history[1])[-1]["content"])
    for tool in mocked_tools:
        tool.assert_not_called()


def test_failure_does_not_skip_other_calls(mocked_tools):
    mocked_tools[0].side_effect = RuntimeError("secret request details")
    client, history = client_for(message([
        call("get_material", '{"formula":"TiO2"}', "a"),
        call("search_papers", '{"query":"HEA"}', "b"),
    ]), message(content="Partial answer"))
    assert "verified evidence" in copilot.ask_material_question("Question", client=client)
    results = tool_results(history[1])
    assert "error" in json.loads(results[0]["content"])
    assert "secret" not in results[0]["content"]
    assert isinstance(json.loads(results[1]["content"]), list)
    mocked_tools[1].assert_called_once()


@pytest.mark.parametrize("result", [object(), {"distance": float("nan")}])
def test_invalid_tool_output_becomes_json_error(result, mocked_tools):
    mocked_tools[0].return_value = result
    assert "error" in json.loads(copilot.execute_tool("get_material", '{"formula":"TiO2"}'))


def test_round_limit_requests_final_answer(monkeypatch, mocked_tools):
    monkeypatch.setattr(copilot, "MAX_TOOL_ROUNDS", 1)
    client, history = client_for(message([call("get_material", '{"formula":"TiO2"}')]),
                                 message(content="Limited answer"))
    assert copilot.ask_material_question("Question", client=client) == "Limited answer"
    assert history[-1]["tool_choice"] == "none"
    mocked_tools[0].assert_called_once()


def test_model_cannot_bypass_round_limit(monkeypatch, mocked_tools):
    monkeypatch.setattr(copilot, "MAX_TOOL_ROUNDS", 1)
    tool_message = message([call("get_material", '{"formula":"TiO2"}')])
    client, _ = client_for(tool_message, tool_message)
    assert "limit reached" in copilot.ask_material_question("Question", client=client)
    mocked_tools[0].assert_called_once()


def test_per_round_limit_still_answers_every_call(monkeypatch, mocked_tools):
    monkeypatch.setattr(copilot, "MAX_CALLS_PER_ROUND", 1)
    client, history = client_for(message([
        call("get_material", '{"formula":"TiO2"}', "a"),
        call("get_material", '{"formula":"Fe"}', "b"),
    ]), message(content="Done"))
    copilot.ask_material_question("Question", client=client)
    assert "error" in json.loads(tool_results(history[1])[-1]["content"])
    assert history[1]["messages"][-1]["tool_call_id"] == "b"
    mocked_tools[0].assert_called_once()


def test_empty_model_answer():
    client, _ = client_for(message())
    assert "no answer" in copilot.ask_material_question("Question", client=client)


@pytest.mark.parametrize("question", ["", " ", None])
def test_invalid_question(question):
    with pytest.raises(ValueError):
        copilot.ask_material_question(question)


def test_grounding_instructions_reach_final_model_turn():
    client, history = client_for(
        message([call("search_papers", '{"query":"dual phase deformation"}')]),
        message(content="The retrieved abstract corpus does not provide enough evidence."),
    )
    copilot.ask_material_question("Explain the slip systems", client=client)
    prompt = history[0]["messages"][0]["content"]
    assert history[-1]["response_format"] == {"type": "json_object"}
    assert "exact contiguous passage" in history[-1]["messages"][0]["content"]
    assert "For literature-based answers, use ONLY facts present in search_papers tool results. Do not supplement them with your own scientific knowledge." in prompt
    for requirement in ("not full papers", "not provide enough evidence", "author placeholders",
                        "Citation metadata must come directly", "slip systems", "numerical values"):
        assert requirement in prompt


def test_citation_fields_are_passed_to_model_unchanged(mocked_tools):
    paper = {"title": "Exact title", "arxiv_url": "https://arxiv.org/abs/2401.00001",
             "arxiv_id": "2401.00001", "authors": ["First Author", "Second Author"],
             "published": "2024-01-01", "abstract": "Only this evidence.", "distance": 0.2}
    mocked_tools[1].return_value = [paper]
    client, history = client_for(message([call("search_papers", '{"query":"HEA"}')]),
                                 message(content="Answer"))
    copilot.ask_material_question("Question", client=client)
    assert json.loads(tool_results(history[-1])[-1]["content"]) == [paper]


@pytest.fixture
def citation_paper():
    return {"arxiv_id": "2401.00001", "title": "Exact retrieved title",
            "arxiv_url": "https://arxiv.org/abs/2401.00001v1",
            "authors": ["Real Retrieved Author"], "published": "2024-01-01",
            "excerpt": "The phases show distinct mechanical properties. No slip system is identified."}


def test_verified_literature_answer(citation_paper, mocked_tools):
    mocked_tools[1].return_value = [citation_paper]
    payload = json.dumps({"evidence": [{"arxiv_id": citation_paper["arxiv_id"],
                                        "quote": citation_paper["excerpt"]}]})
    client, _ = client_for(message([call("search_papers", '{"query":"HEA"}')]),
                           message(content=payload))
    answer = copilot.ask_material_question("Mechanisms?", client=client)
    assert citation_paper["excerpt"] in answer
    assert f'[{citation_paper["title"]}]({citation_paper["arxiv_url"]})' in answer
    assert "not full papers" in answer
    assert "does not provide enough evidence" in answer


@pytest.mark.parametrize("payload", [
    {"evidence": [{"arxiv_id": "fake-id", "quote": "The phases show distinct mechanical properties."}]},
    {"evidence": [{"arxiv_id": "2401.00001", "quote": "Dislocation slip dominates deformation."}]},
    {"evidence": [{"arxiv_id": "2401.00001", "quote": "The phases show distinct mechanical properties.",
                   "authors": "A. et al."}]},
    {"evidence": [], "explanation": "Unsupported mechanism"},
    {"evidence": []},
])
def test_unverified_claims_and_citations_are_not_rendered(payload, citation_paper):
    answer = copilot.render_literature_answer(json.dumps(payload), {citation_paper["arxiv_id"]: citation_paper})
    assert "No unverified scientific explanation" in answer
    assert "A. et al." not in answer
    assert "Dislocation slip" not in answer


def test_freeform_hallucination_rejected(citation_paper):
    answer = copilot.render_literature_answer("Dislocation slip dominates (A. et al.).", {citation_paper["arxiv_id"]: citation_paper})
    assert "No unverified scientific explanation" in answer


def test_mixed_answer_preserves_computed_values(citation_paper, mocked_tools):
    mocked_tools[1].return_value = [citation_paper]
    payload = json.dumps({"evidence": [{"arxiv_id": citation_paper["arxiv_id"],
                                        "quote": citation_paper["excerpt"]}]})
    client, _ = client_for(message([
        call("get_material", '{"formula":"TiO2"}', "a"),
        call("search_papers", '{"query":"HEA"}', "b"),
    ]), message(content=payload))
    answer = copilot.ask_material_question("Mixed question", client=client)
    assert citation_paper["excerpt"] in answer
    assert '"band_gap": 2.06' in answer
    assert "not necessarily experimental" in answer


def test_whitespace_and_unicode_quote_matching(citation_paper):
    citation_paper['excerpt'] = "The alloy’s two‑phase structure\nshows   strength."
    payload = json.dumps({'evidence': [{'arxiv_id': citation_paper['arxiv_id'],
                                     'quote': "The alloy's two-phase structure shows strength."}]})
    evidence, reason = copilot.validate_evidence(payload, {citation_paper['arxiv_id']: citation_paper})
    assert reason == 'successful_grounded_evidence'
    assert evidence


@pytest.mark.parametrize('bad,reason', [
    ('not JSON', 'malformed_evidence_json'),
    (json.dumps({'evidence': [{'arxiv_id': 'fake', 'quote': 'text'}]}), 'unknown_arxiv_id'),
    (json.dumps({'evidence': [{'arxiv_id': '2401.00001', 'quote': 'Invented science'}]}), 'quotation_mismatch'),
    (json.dumps({'evidence': []}), 'no_evidence_selected'),
])
def test_retry_once_with_validation_feedback(bad, reason, citation_paper, mocked_tools):
    mocked_tools[1].return_value = [citation_paper]
    good = json.dumps({'evidence': [{'arxiv_id': citation_paper['arxiv_id'], 'quote': citation_paper['excerpt']}]})
    client, history = client_for(message([call('search_papers', '{"query":"HEA"}')]),
                                 message(content=bad), message(content=good))
    diagnostics = []
    answer = copilot.ask_material_question('HEA?', client=client, evidence_trace=diagnostics)
    assert citation_paper['title'] in answer
    assert [d['reason'] for d in diagnostics] == [reason, 'successful_grounded_evidence']
    assert reason in history[-1]['messages'][-1]['content']
    assert len(history) == 3
    assert reason not in answer


def test_failed_retry_retains_computed_properties(citation_paper, mocked_tools):
    mocked_tools[1].return_value = [citation_paper]
    client, history = client_for(message([
        call('get_material', '{"formula":"TiO2"}', 'a'),
        call('search_papers', '{"query":"HEA"}', 'b'),
    ]), message(content='bad'), message(content='bad again'))
    diagnostics = []
    answer = copilot.ask_material_question('Mixed?', client=client, evidence_trace=diagnostics)
    assert '"band_gap": 2.06' in answer
    assert 'No unverified scientific explanation' in answer
    assert len(history) == 3
    assert [d['reason'] for d in diagnostics] == ['malformed_evidence_json'] * 2


def test_cited_insufficient_crystallographic_detail(citation_paper):
    citation_paper['excerpt'] = 'Dislocations within the two phases were studied by microscopy.'
    payload = json.dumps({'evidence': [{'arxiv_id': citation_paper['arxiv_id'], 'quote': citation_paper['excerpt']}]})
    answer = copilot.render_literature_answer(payload, {citation_paper['arxiv_id']: citation_paper},
                                             'What slip systems and Burgers vectors are reported?')
    assert 'does not specify slip-system indices or Burgers vectors' in answer
    assert citation_paper['arxiv_url'] in answer
    assert citation_paper['excerpt'] in answer


@pytest.mark.parametrize('formula', ['XYZ123', 'NotARealMaterial'])
def test_invalid_formula_rejection_policy_without_lookup(formula, mocked_tools):
    answer = f'{formula} is not a valid chemical formula. Please provide a valid formula.'
    client, history = client_for(message(content=answer))
    trace = []
    assert copilot.ask_material_question(f'Look up {formula}', client=client, tool_trace=trace) == answer
    assert trace == []
    mocked_tools[0].assert_not_called()
    assert 'WITHOUT get_material' in history[0]['messages'][0]['content']
    assert 'unless a lookup was actually performed' in history[0]['messages'][0]['content']
    assert 'no entry' not in answer


def test_normalization_does_not_change_scientific_values(citation_paper):
    citation_paper['excerpt'] = 'Stress was −10 MPa, not 10 MPa.'
    payload = json.dumps({'evidence': [{'arxiv_id': citation_paper['arxiv_id'], 'quote': 'Stress was 10 MPa'}]})
    assert copilot.validate_evidence(payload, {citation_paper['arxiv_id']: citation_paper})[1] == 'quotation_mismatch'
