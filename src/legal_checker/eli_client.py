"""Client of the Sejm ELI API: act metadata and content (SPEC Task 2.2)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
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


class ActContent(BaseModel):
    """Raw act content downloaded from a ``TextSource``."""

    source: TextSource
    url: str  # absolute URL of the downloaded file, used as ``source_url``
    media_type: Literal["application/pdf", "text/html"]
    data: bytes


class FetchedAct(BaseModel):
    """Act metadata and current-wording content, as fetched at ``fetched_at``."""

    metadata: ActMetadata
    content: ActContent
    fetched_at: datetime  # UTC; its date is the act's ``as_of_date``


class ActCache:
    """Two-level cache of fetched acts: a dict in RAM over files in ``<cache_dir>/eli/``.

    Each act gets a directory (e.g. ``DU_2014_827/``) with ``act.json`` (metadata, source,
    fetch time) and the raw ``content.pdf`` / ``content.html``. Entries older than
    ``max_age`` count as misses, so the act's current wording is re-fetched;
    ``max_age=None`` never expires (e.g. offline test runs on a prepared cache).

    Data derived from an act (e.g. its extracted text) is stored next to it as
    ``derived-<name>`` files. It is only served while the act entry is valid and is
    deleted whenever the act is stored again.
    """

    def __init__(self, cache_dir: Path, max_age: timedelta | None = timedelta(days=1)):
        self.root = Path(cache_dir) / "eli"
        self.max_age = max_age
        self._memory: dict[str, FetchedAct] = {}
        self._derived: dict[tuple[str, str], str] = {}

    def get(self, ref: EliRef) -> FetchedAct | None:
        act = self._memory.get(ref.key) or self._read(ref)
        if act is None or self._expired(act):
            return None
        self._memory[ref.key] = act
        return act

    def put(self, ref: EliRef, act: FetchedAct) -> None:
        directory = self._dir(ref)
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("derived-*"):
            stale.unlink()
        self._derived = {k: v for k, v in self._derived.items() if k[0] != ref.key}
        (directory / self._content_file(act.content)).write_bytes(act.content.data)
        # act.json is written last and atomically: its presence marks a complete entry.
        tmp = directory / "act.json.tmp"
        tmp.write_text(act.model_dump_json(by_alias=True, exclude={"content": {"data"}}, indent=2), encoding="utf-8")
        tmp.replace(directory / "act.json")
        self._memory[ref.key] = act

    def get_derived(self, ref: EliRef, name: str) -> str | None:
        """Return data derived from the act under ``name``, if the act entry is still valid."""
        if self.get(ref) is None:
            return None
        if (ref.key, name) not in self._derived:
            try:
                self._derived[(ref.key, name)] = (self._dir(ref) / f"derived-{name}").read_text(encoding="utf-8")
            except OSError:
                return None
        return self._derived[(ref.key, name)]

    def put_derived(self, ref: EliRef, name: str, data: str) -> None:
        """Store data derived from the act under ``name``; a no-op if the act is not cached."""
        if self.get(ref) is None:
            return
        tmp = self._dir(ref) / f"derived-{name}.tmp"
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(self._dir(ref) / f"derived-{name}")
        self._derived[(ref.key, name)] = data

    def _read(self, ref: EliRef) -> FetchedAct | None:
        directory = self._dir(ref)
        try:
            data = json.loads((directory / "act.json").read_text(encoding="utf-8"))
            content = ActContent.model_validate({**data["content"], "data": b""})
            data["content"]["data"] = (directory / self._content_file(content)).read_bytes()
            return FetchedAct.model_validate(data)
        except (OSError, ValueError, KeyError):  # missing or corrupted entry: treat as a miss
            return None

    def _expired(self, act: FetchedAct) -> bool:
        return self.max_age is not None and datetime.now(UTC) - act.fetched_at > self.max_age

    def _dir(self, ref: EliRef) -> Path:
        return self.root / ref.key.replace("/", "_")

    @staticmethod
    def _content_file(content: ActContent) -> str:
        return "content.html" if content.media_type == "text/html" else "content.pdf"


class EliClient:
    """Synchronous ELI API client with timeouts, connection retries and an optional act cache."""

    def __init__(self, base_url: str, http: httpx.Client | None = None, cache: ActCache | None = None):
        self.http = http or httpx.Client(
            base_url=base_url,
            timeout=_TIMEOUT,
            transport=httpx.HTTPTransport(retries=_CONNECT_RETRIES),
        )
        self.cache = cache

    @classmethod
    def from_settings(cls, settings: Settings) -> "EliClient":
        return cls(settings.eli_api_base_url, cache=ActCache(settings.cache_dir))

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

    def download_text(self, source: TextSource) -> ActContent:
        """Download the file a ``TextSource`` points to and check its media type."""
        response = self._get(source.path)
        media_type = "text/html" if source.kind == "html" else "application/pdf"
        received = response.headers.get("content-type", "").split(";")[0].strip()
        if received != media_type:
            raise EliClientError(f"API ELI zwróciło {received or 'brak typu'} zamiast {media_type} dla {source.path}")
        return ActContent(source=source, url=str(response.url), media_type=media_type, data=response.content)

    def fetch_act(self, ref: EliRef) -> FetchedAct:
        """Fetch metadata and current-wording content of an act, using the cache if set."""
        if self.cache and (cached := self.cache.get(ref)):
            return cached
        metadata = self.get_metadata(ref)
        content = self.download_text(self.select_text_source(metadata))
        act = FetchedAct(metadata=metadata, content=content, fetched_at=datetime.now(UTC))
        if self.cache:
            self.cache.put(ref, act)
        return act

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
