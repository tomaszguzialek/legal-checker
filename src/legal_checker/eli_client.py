"""Client of the Sejm ELI API: act metadata and content (SPEC Task 2.2)."""

from typing import Literal

import httpx
from pydantic import BaseModel, Field

from legal_checker.config import Settings
from legal_checker.models import EliRef

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_CONNECT_RETRIES = 2

# Text variant types in ActMetadata.texts. Only "U" (Kancelaria Sejmu's unified text)
# carries the current wording of an amended act; "H" (text.html), "O"/"I" and "T" hold
# the wording as originally published. See docs/adr/0002-current-act-text-source.md.
_UNIFIED = "U"
_CONSOLIDATED_TEXT_RELATION = "Inf. o tekście jednolitym"
_CONSOLIDATED_TEXT_TYPES = ("U", "T", "O")


class EliClientError(RuntimeError):
    """The ELI API request failed or returned an unexpected response."""


class ActText(BaseModel):
    """One text variant of an act, e.g. ``H`` (HTML) or ``U`` (unified text PDF)."""

    file_name: str = Field(alias="fileName")
    type: str


class ActMetadata(BaseModel):
    """Subset of ``GET /acts/{publisher}/{year}/{position}`` used by the pipeline."""

    eli: str = Field(alias="ELI")
    title: str
    type: str
    status: str
    in_force: str | None = Field(None, alias="inForce")  # "IN_FORCE" / "NOT_IN_FORCE"
    promulgation: str | None = None
    entry_into_force: str | None = Field(None, alias="entryIntoForce")
    change_date: str | None = Field(None, alias="changeDate")
    text_html: bool = Field(False, alias="textHTML")
    text_pdf: bool = Field(False, alias="textPDF")
    texts: list[ActText] = Field(default_factory=list)
    # Relation name (e.g. "Inf. o tekście jednolitym") -> referenced ELI keys.
    references: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict) -> "ActMetadata":
        references = {
            relation: [entry["id"] for entry in entries]
            for relation, entries in data.get("references", {}).items()
        }
        return cls.model_validate({**data, "references": references})


class TextSource(BaseModel):
    """Where to download the most current wording of an act from."""

    eli: str  # act owning the file: the act itself or its consolidated-text notice
    kind: Literal["unified_pdf", "consolidated_pdf", "html", "pdf"]
    path: str  # API path relative to the base URL


class EliClient:
    """Synchronous ELI API client with timeouts and connection retries."""

    def __init__(self, base_url: str, http: httpx.Client | None = None):
        self.http = http or httpx.Client(
            base_url=base_url,
            timeout=_TIMEOUT,
            transport=httpx.HTTPTransport(retries=_CONNECT_RETRIES),
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "EliClient":
        return cls(settings.eli_api_base_url)

    def __enter__(self) -> "EliClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self.http.close()

    def get_metadata(self, ref: EliRef) -> ActMetadata:
        """Fetch act metadata: title, status, dates, text variants and references."""
        response = self._get(f"/acts/{ref.key}")
        return ActMetadata.from_api(response.json())

    def select_text_source(self, meta: ActMetadata) -> TextSource:
        """Pick the source of the act's current wording ("na dzień dzisiejszy").

        Order: the act's unified text PDF, then the newest consolidated text announced
        for it, then HTML, then the main PDF. HTML and the main PDF hold the wording as
        published, which is current only for acts that were never amended.
        """
        unified = next((t for t in meta.texts if t.type == _UNIFIED), None)
        if unified:
            return TextSource(eli=meta.eli, kind="unified_pdf", path=f"/acts/{meta.eli}/text/U/{unified.file_name}")

        consolidated = meta.references.get(_CONSOLIDATED_TEXT_RELATION, [])
        if consolidated:
            newest = max(map(EliRef.from_key, consolidated), key=lambda ref: (ref.year, ref.position))
            notice = self.get_metadata(newest)
            for text_type in _CONSOLIDATED_TEXT_TYPES:
                text = next((t for t in notice.texts if t.type == text_type), None)
                if text:
                    path = f"/acts/{newest.key}/text/{text_type}/{text.file_name}"
                    return TextSource(eli=newest.key, kind="consolidated_pdf", path=path)

        if meta.text_html:
            return TextSource(eli=meta.eli, kind="html", path=f"/acts/{meta.eli}/text.html")
        if meta.text_pdf:
            return TextSource(eli=meta.eli, kind="pdf", path=f"/acts/{meta.eli}/text.pdf")
        raise EliClientError(f"Akt {meta.eli} nie ma dostępnej treści w API ELI")

    def _get(self, path: str) -> httpx.Response:
        try:
            response = self.http.get(path)
        except httpx.HTTPError as exc:
            raise EliClientError(f"Błąd połączenia z API ELI ({path}): {exc}") from exc
        if response.status_code == 404:
            raise EliClientError(f"Nie znaleziono zasobu w API ELI: {path}")
        if response.is_error:
            raise EliClientError(f"API ELI zwróciło błąd {response.status_code} dla {path}")
        return response
