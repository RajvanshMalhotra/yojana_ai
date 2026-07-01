from server import _build_prompt, _build_messages


CHUNKS = [{"title": "PM-KISAN", "benefit_summary": "Cash", "eligibility_summary": "Farmers", "text": "Details here"}]
WEB    = [{"title": "Startup Grant", "snippet": "Rs 10L for startups", "url": "https://example.com"}]


def test_build_prompt_no_web():
    result = _build_prompt("schemes for farmers", CHUNKS)
    assert "CONTEXT:" in result
    assert "[1] PM-KISAN" in result
    assert "WEB RESULTS" not in result


def test_build_prompt_with_web():
    result = _build_prompt("schemes for AI startups", CHUNKS, WEB)
    assert "WEB RESULTS:" in result
    assert "[W1] Startup Grant" in result
    assert "Rs 10L for startups" in result
    assert "https://example.com" in result


def test_build_messages_without_web_has_no_W_cite_instruction():
    msgs = _build_messages("prompt", [], "", has_web=False)
    assert "[W1]" not in msgs[0]["content"]


def test_build_messages_with_web_has_W_cite_instruction():
    msgs = _build_messages("prompt", [], "", has_web=True)
    assert "[W1]" in msgs[0]["content"] or "W1" in msgs[0]["content"]
