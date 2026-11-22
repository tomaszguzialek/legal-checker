import io
import json

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from legal_checker import cli
from legal_checker.agent import graph as graph_module
from legal_checker.config import Settings
from legal_checker.eli_resolver import ActResolutionError
from legal_checker.models import ComplianceStatus, LlmEvaluationPayload

from conftest import ELI_TEST_URL

WITHDRAWAL = "Klient może odstąpić od umowy zawartej na odległość w terminie 14 dni bez podania przyczyny."
PAYLOAD = LlmEvaluationPayload(
    status=ComplianceStatus.COMPLIANT,
    confidence=0.9,
    document_excerpt="w terminie 14 dni",
    matched_provision="art. 27",
    rationale="Zapis powtarza ustawowy termin odstąpienia.",
)


def get_provision(article: str, call_id: str) -> dict:
    return {"name": "get_provision", "args": {"article": article}, "id": call_id}


@pytest.fixture
def fake_models(cached_act, tmp_path, monkeypatch):
    """Run the CLI on the cached consumer rights act and fake models.

    The research model asks for art. 30 and then, with a budget of one call, for art. 31, which is not run.
    """
    settings = Settings(eli_api_base_url=ELI_TEST_URL, cache_dir=tmp_path, agent_max_tool_calls=1)

    def fake_build_models(_settings, _tools):
        research = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage("", tool_calls=[get_provision("30", "c1")]),
                    AIMessage("", tool_calls=[get_provision("31", "c2")]),
                ]
            )
        )
        evaluation = RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": PAYLOAD, "parsing_error": None})
        return research, evaluation

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(graph_module, "build_models", fake_build_models)


def split_output(out: str) -> tuple[dict, str]:
    """The JSON result printed first, and the report after it."""
    result, end = json.JSONDecoder().raw_decode(out)
    return result, out[end:]


def test_prints_the_json_result_and_a_report(fake_models, capsys):
    exit_code = cli.main(["--text", WITHDRAWAL, "--hint", "DU/2014/827"])

    assert exit_code == 0
    result, report = split_output(capsys.readouterr().out)
    assert result["status"] == "ZGODNY"
    assert result["legal_basis"]["eli"] == "DU/2014/827"
    assert result["consulted_units"][-1] == "DU/2014/827:art.30"
    assert "ZGODNY" in report
    assert "Zapis powtarza ustawowy termin odstąpienia." in report
    assert "  • DU/2014/827:art.30" in report
    assert "Wywołania narzędzi agenta (2):" in report
    assert "get_provision(article='30') → DU/2014/827:art.30" in report
    assert f"get_provision(article='31') → {cli.NOT_RUN}" in report
    assert "Ostrzeżenie ekstrakcji" in report


def test_reads_the_fragment_from_a_file(fake_models, tmp_path, capsys):
    fragment = tmp_path / "fragment.txt"
    fragment.write_text(WITHDRAWAL, encoding="utf-8")

    exit_code = cli.main(["--file", str(fragment), "--hint", "Dz.U. z 2014 r. poz. 827"])

    assert exit_code == 0
    result, _ = split_output(capsys.readouterr().out)
    assert result["document_excerpt"] == "w terminie 14 dni"


def test_reads_the_fragment_from_stdin(fake_models, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(WITHDRAWAL))

    assert cli.main(["--file", "-", "--hint", "DU/2014/827"]) == 0
    result, _ = split_output(capsys.readouterr().out)
    assert result["status"] == "ZGODNY"


def test_text_and_file_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--text", WITHDRAWAL, "--file", "fragment.txt", "--hint", "DU/2014/827"])

    assert exit_info.value.code == 2


def test_hint_is_required(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--text", WITHDRAWAL])

    assert exit_info.value.code == 2


def test_unresolvable_hint_is_reported_without_a_traceback(monkeypatch, capsys):
    def failing_run(request, settings):
        raise ActResolutionError("Nie znaleziono aktu prawnego pasującego do wskazówki: 'xyz'")

    monkeypatch.setattr(cli, "run", failing_run)
    monkeypatch.setattr(cli, "get_settings", Settings)

    assert cli.main(["--text", WITHDRAWAL, "--hint", "xyz"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Błąd: Nie znaleziono aktu prawnego" in captured.err


def test_too_short_fragment_is_reported_as_an_error(capsys):
    assert cli.main(["--text", "krótki", "--hint", "DU/2014/827"]) == 1
    assert "document_fragment" in capsys.readouterr().err


def test_missing_file_is_reported_as_an_error(tmp_path, capsys):
    assert cli.main(["--file", str(tmp_path / "missing.txt"), "--hint", "DU/2014/827"]) == 1
    assert "Błąd:" in capsys.readouterr().err
