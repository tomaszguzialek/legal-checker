"""Client of the Sejm ELI API: act metadata and content (SPEC Task 2.2)."""

import httpx
from pydantic import BaseModel, Field

from legal_checker.config import Settings
from legal_checker.models import EliRef

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_CONNECT_RETRIES = 2


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
