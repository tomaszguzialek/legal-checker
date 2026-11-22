from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from legal_checker.agent.nodes import (
    SHORT_CIRCUIT_CONFIDENCE,
    make_act_nodes,
    route_after_retrieve,
    short_circuit_na,
    validate,
)
from legal_checker.bm25_retriever import LateBoundBm25Index
from legal_checker.eli_client import ActCache, ActContent, ActMetadata, EliClient, FetchedAct, TextSource
from legal_checker.models import (
    ActDocument,
    CheckRequest,
    ComplianceStatus,
    EliRef,
    LegalUnit,
    LlmEvaluationPayload,
)
from legal_checker.text_extractor import EXTRACTOR_VERSION, ExtractedText

# Mocked host: ".test" is reserved (RFC 2606), so these tests never reach the real ELI API.
BASE_URL = "https://eli-api.test/eli"
ACT_TEXT = (Path(__file__).parent / "fixtures" / "acts" / "DU_2014_827.txt").read_text(encoding="utf-8")
CONSUMER_RIGHTS = EliRef(publisher="DU", year=2014, position=827)
PDF_URL = f"{BASE_URL}/acts/DU/2014/827/text/U/D20140827Lj.pdf"
FETCHED_AT = datetime(2026, 11, 20, 8, 30, tzinfo=UTC)
WITHDRAWAL = "Klient może odstąpić od umowy zawartej na odległość w terminie 14 dni bez podania przyczyny."


@pytest.fixture
def eli(tmp_path):
    """Client over a never-expiring cache holding the act and its extracted fixture text: no HTTP at all."""
    cache = ActCache(tmp_path, max_age=None)
    source = TextSource(eli="DU/2014/827", kind="unified_pdf", path="/acts/DU/2014/827/text/U/D20140827Lj.pdf")
    content = ActContent(source=source, url=PDF_URL, media_type="application/pdf", data=b"%PDF-1.7 fake")
    metadata = ActMetadata(
        ELI="DU/2014/827",
        title="Ustawa z dnia 30 maja 2014 r. o prawach konsumenta",
        type="Ustawa",
        status="obowiązujący",
    )
    cache.put(CONSUMER_RIGHTS, FetchedAct(metadata=metadata, content=content, fetched_at=FETCHED_AT))
    extracted = ExtractedText(text=ACT_TEXT, warnings=["Ostrzeżenie ekstrakcji"])
    cache.put_derived(CONSUMER_RIGHTS, f"extracted-v{EXTRACTOR_VERSION}.json", extracted.model_dump_json())
    with EliClient(BASE_URL, cache=cache) as client:
        yield client


def run_act_nodes(eli: EliClient, fragment: str, index: LateBoundBm25Index | None = None) -> dict:
    """Run the deterministic nodes in graph order, merging updates (warnings are appended)."""
    nodes = make_act_nodes(eli, index or LateBoundBm25Index(), top_k=5)
    state: dict = {"request": CheckRequest(document_fragment=fragment, act_hint="Dz.U. z 2014 r. poz. 827")}
    for name in ["resolve_act", "fetch_act", "chunk_act", "retrieve"]:
        update = nodes[name](state)
        state = {**state, **update, "warnings": state.get("warnings", []) + update.get("warnings", [])}
    return state


def test_act_nodes_resolve_fetch_chunk_and_retrieve(eli):
    index = LateBoundBm25Index()

    state = run_act_nodes(eli, WITHDRAWAL, index)

    assert state["eli_ref"] == CONSUMER_RIGHTS
    assert state["act"] == ActDocument(
        eli="DU/2014/827",
        title="Ustawa z dnia 30 maja 2014 r. o prawach konsumenta",
        source_url=PDF_URL,
        as_of_date="2026-11-20",
        text=ACT_TEXT,
    )
    assert state["units"] and all(unit.eli == "DU/2014/827" for unit in state["units"])
    assert state["warnings"] == ["Ostrzeżenie ekstrakcji"]
    assert "DU/2014/827:art.27" in [unit.unit_id for unit, _ in state["candidates"]]
    assert index.get("27").unit_id == "DU/2014/827:art.27"  # the tools' index is built
    assert route_after_retrieve(state) == "research"


def test_late_bound_index_fails_clearly_before_retrieve():
    with pytest.raises(RuntimeError, match="not built yet"):
        LateBoundBm25Index().search("odstąpienie")


def test_no_shared_token_short_circuits_to_not_applicable(eli):
    state = run_act_nodes(eli, "Qwerty zxcvb asdfgh 2099.")

    assert state["candidates"] == []
    assert route_after_retrieve(state) == "short_circuit_na"

    result = short_circuit_na(state)["result"]

    assert result.status == ComplianceStatus.NOT_APPLICABLE
    assert result.confidence == SHORT_CIRCUIT_CONFIDENCE
    assert result.document_excerpt is None and result.legal_basis.provision is None
    assert result.legal_basis.eli == "DU/2014/827"
    assert result.legal_basis.source_url == PDF_URL
    assert result.consulted_units == []
    assert result.warnings == ["Ostrzeżenie ekstrakcji"]


def test_validate_attaches_metadata_and_everything_the_agent_saw(eli):
    state = run_act_nodes(eli, WITHDRAWAL)
    unit = LegalUnit(unit_id="DU/2014/827:art.38", eli="DU/2014/827", article="38", hierarchy_path=[], text="Art. 38.")
    candidate_ids = [unit.unit_id for unit, _ in state["candidates"]]
    state["messages"] = [
        AIMessage("", tool_calls=[{"name": "get_provision", "args": {"article": "38"}, "id": "c1"}]),
        ToolMessage("[DU/2014/827:art.38] Art. 38.", tool_call_id="c1", artifact=[unit.unit_id, candidate_ids[0]]),
    ]
    state["llm_payload"] = LlmEvaluationPayload(
        status=ComplianceStatus.COMPLIANT,
        confidence=0.85,
        document_excerpt="w terminie 14 dni",
        matched_provision="art. 27",
        rationale="Zapis powtarza ustawowy termin.",
    )

    result = validate(state)["result"]

    assert result.status == ComplianceStatus.COMPLIANT and result.confidence == 0.85
    assert result.legal_basis.model_dump() == {
        "act_title": "Ustawa z dnia 30 maja 2014 r. o prawach konsumenta",
        "eli": "DU/2014/827",
        "provision": "art. 27",
        "as_of_date": "2026-11-20",
        "source_url": PDF_URL,
    }
    assert result.consulted_units == [*candidate_ids, "DU/2014/827:art.38"]
