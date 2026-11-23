"""Domain data contracts (SPEC section 3)."""

from enum import Enum

from pydantic import BaseModel, Field


class CheckRequest(BaseModel):
    document_fragment: str = Field(..., min_length=10, description="Fragment dokumentu firmowego po polsku")
    act_hint: str = Field(..., min_length=2, description="Wskazówka aktu: ELI, link, sygnatura Dz.U./M.P. lub tytuł")


class EliRef(BaseModel):
    """Canonical ELI key of an act, e.g. ``DU/2014/827``."""

    publisher: str  # "DU" (Dziennik Ustaw) or "MP" (Monitor Polski)
    year: int
    position: int

    @property
    def key(self) -> str:
        return f"{self.publisher}/{self.year}/{self.position}"

    @classmethod
    def from_key(cls, key: str) -> "EliRef":
        publisher, year, position = key.split("/")
        return cls(publisher=publisher, year=int(year), position=int(position))


class ActDocument(BaseModel):
    """Fetched act: source metadata plus the cleaned text."""

    eli: str
    title: str
    source_url: str
    as_of_date: str  # ISO YYYY-MM-DD (fetch date / current legal state)
    text: str


class LegalUnit(BaseModel):
    unit_id: str  # e.g. "DU/2014/827:art.27"
    eli: str  # e.g. "DU/2014/827"
    article: str  # e.g. "27"
    paragraph: str | None = None  # set when long articles are split into paragraphs
    hierarchy_path: list[str]  # e.g. ["Rozdział 4", "Art. 27"]
    text: str  # cleaned text of the unit


class ComplianceStatus(str, Enum):
    COMPLIANT = "ZGODNY"
    PARTIALLY_COMPLIANT = "CZĘŚCIOWO ZGODNY"
    NON_COMPLIANT = "NIEZGODNY"
    NOT_APPLICABLE = "NIE DOTYCZY / BRAK PODSTAW DO OCENY"


class LlmEvaluationPayload(BaseModel):
    """Fields the LLM fills in; enforced via JSON Schema structured output."""

    status: ComplianceStatus
    applicability_unclear: bool = Field(
        ...,
        description="Czy z fragmentu nie wynika, czy akt w ogóle ma zastosowanie do stron (np. konsument czy przedsiębiorca)",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    document_excerpt: str | None = Field(
        None,
        description="Dosłowny cytat problematycznego (lub ocenianego) fragmentu z dokumentu wejściowego",
    )
    matched_provision: str | None = Field(
        None,
        description="Konkretny przepis z dostarczonego kontekstu, np. 'art. 27' lub 'art. 12 ust. 1 pkt 2'",
    )
    rationale: str = Field(..., description="Zwięzłe uzasadnienie oceny w 2-5 zdaniach po polsku")
    suggested_fix: str | None = Field(None, description="Opcjonalna propozycja poprawki brzmienia zapisu")


class LegalBasis(BaseModel):
    act_title: str
    eli: str
    provision: str | None
    as_of_date: str  # ISO YYYY-MM-DD (date of the assessment: the law as it stands that day)
    source_url: str


class ComplianceResult(BaseModel):
    """Final result: LLM assessment plus source metadata attached by code."""

    status: ComplianceStatus
    confidence: float
    document_excerpt: str | None
    legal_basis: LegalBasis
    rationale: str
    suggested_fix: str | None
    consulted_units: list[str] = Field(default_factory=list)  # unit_ids of provisions the agent saw
    warnings: list[str] = Field(default_factory=list)
