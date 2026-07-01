from unittest.mock import MagicMock, patch
import importlib


def _make_mock_client(response_text: str):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value.choices = [choice]
    return client


def test_judge_returns_sufficient():
    import server
    with patch.dict(server._state, {"gen_client": _make_mock_client("sufficient")}):
        result = server._judge(
            [{"title": "PM-KISAN", "benefit_summary": "Cash to farmers"}],
            "schemes for farmers"
        )
    assert result == "sufficient"


def test_judge_returns_insufficient():
    import server
    with patch.dict(server._state, {"gen_client": _make_mock_client("insufficient")}):
        result = server._judge([], "schemes for AI drones")
    assert result == "insufficient"


def test_judge_treats_unexpected_as_insufficient():
    import server
    with patch.dict(server._state, {"gen_client": _make_mock_client("I don't know")}):
        result = server._judge(
            [{"title": "Any", "benefit_summary": "Anything"}],
            "some query"
        )
    assert result == "insufficient"
