# Specyfikacja Implementacyjna (PoC): AI Legal & Compliance Checker

**Wersja:** 1.1 (PoC — warstwa agentowa na LangChain / LangGraph)  
**Tryb pracy:** Spec-Driven Development (do etapowej realizacji w Claude Code)  
**Zakres czasowy oceny:** Wyłącznie stan prawny „na dzień dzisiejszy” (bez silnika osi czasu i historycznych dat wstecznych).

---

## 1. Cel i założenia architektoniczne (Context & Guardrails)

### 1.1 Cel prototypu
Zbudowanie działającego od końca do końca (*end-to-end*) narzędzia CLI / lekkiego API w Pythonie, które:
1. Przyjmuje fragment dokumentu firmowego w języku polskim oraz wskazówkę aktu prawnego (*wariant minimalny*).
2. Deterministycznie rozwiązuje wskazówkę na kanoniczny identyfikator **ELI** (`{publisher}/{year}/{position}`, np. `DU/2014/827`) i pobiera aktualną treść aktu z publicznego **API ELI Sejmu** (`https://api.sejm.gov.pl/eli`).
3. Czyści treść i tnie ją **po strukturze prawnej** (artykuły, ustępy, punkty) z zachowaniem hierarchii nadrzędnej.
4. Filtruje kandydatów w pamięci operacyjnej (**RAM**) za pomocą algorytmu **BM25** (bez zewnętrznych baz danych i wektorów).
5. Orkiestruje cały przepływ jako **graf stanów LangGraph** (`StateGraph`), w którym agent LLM (LangChain) może za pomocą **narzędzi tylko do odczytu** (*read-only tools*) dociągnąć dodatkowe przepisy z indeksu BM25, zanim wyda ocenę.
6. Ocenia zgodność fragmentu z zebranymi przepisami za pomocą **LLM z wymuszonym Structured Outputs (JSON Schema)** (`with_structured_output`) oraz przeprowadza deterministyczną walidację wyniku w kodzie.

### 1.2 Podział odpowiedzialności: Kod deterministyczny vs. Orkiestracja vs. LLM
* **Kod deterministyczny (Python):**
  * Normalizacja i trójstopniowe parsowanie wskazówki aktu na klucz ELI.
  * Komunikacja z API ELI, wybór najbardziej aktualnego tekstu (np. tekst ujednolicony/jednolity), ekstrakcja tekstu z PDF/HTML i czyszczenie nagłówków/stopek.
  * Parsowanie struktury prawnej (budowa drzewa jednostek redakcyjnych).
  * Indeksowanie i wyszukiwanie kandydatów w pamięci za pomocą **BM25** + prosty cache w RAM / na dysku lokalnym.
  * Implementacja narzędzi agenta — narzędzia są cienkimi, deterministycznymi wrapperami na indeks BM25 i listę `LegalUnit` (bez dostępu do sieci, bez efektów ubocznych).
  * Wstrzykiwanie twardych metadanych źródłowych (`source_url`, `as_of_date`, `eli`, `act_title`) oraz walidacja post-LLM (sprawdzenie, czy cytat pochodzi z wejścia, czy wskazany przepis był faktycznie widziany przez agenta, walidacja schematu).
* **Orkiestracja (LangGraph):**
  * Graf stanów z jawnie zdefiniowanymi węzłami (*nodes*) i krawędziami warunkowymi (*conditional edges*): krótkie spięcie dla wyniku BM25 = 0, pętla agent ↔ narzędzia z limitem wywołań, ponowienie oceny po błędzie schematu.
  * Węzły deterministyczne (resolver, klient ELI, chunker, retriever, post-validator) są zwykłymi funkcjami Pythona operującymi na stanie grafu — LLM nie decyduje o ich kolejności.
  * Twarde limity: `AGENT_MAX_TOOL_CALLS` oraz `recursion_limit` przy wywołaniu grafu.
* **Model językowy (LLM, przez LangChain `BaseChatModel`):**
  * **Etap research (agentowy):** decyzja, czy kandydaci z BM25 wystarczają, czy należy dociągnąć dodatkowe przepisy (np. artykuł przywołany w treści kandydata) za pomocą narzędzi `search_provisions` / `get_provision`.
  * **Etap oceny:** wyłącznie merytoryczna ocena zgodności na podstawie zebranego tekstu przepisów i fragmentu dokumentu, wskazanie konkretnej jednostki redakcyjnej (`art.`, `ust.`, `pkt`), zacytowanie problematycznego fragmentu dokumentu oraz wygenerowanie zwięzłego uzasadnienia (2–5 zdań) i opcjonalnej propozycji poprawki.

### 1.3 Świadome uproszczenia w PoC (Out of Scope)
* Brak obsługi dat wstecznych (`as_of_date` ustawiane automatycznie na bieżącą datę pobrania).
* Brak automatycznego zgadywania aktu prawnego bez żadnej wskazówki (*wariant ambitny* odłożony na etap produkcyjny).
* Brak bazy danych (PostgreSQL), bazy wektorowej (`pgvector`/Qdrant) i embeddingów — wszystko działa w pamięci operacyjnej procesu (z opcjonalnym lokalnym cache plików `.json` w katalogu `.cache/`).
* Brak systemu wieloagentowego (*multi-agent*, *supervisor*) i Deep Agents — jeden agent z dwoma narzędziami tylko do odczytu w jawnym `StateGraph` (uzasadnienie: [ADR 0001](../docs/adr/0001-langgraph-stategraph-over-deep-agents.md)).
* Brak narzędzi z dostępem do sieci dla agenta (agent nie wywołuje API ELI ani wyszukiwarki internetowej — pobranie aktu jest deterministycznym węzłem grafu).
* Brak trwałego checkpointera LangGraph (stan żyje tylko w trakcie jednego wywołania), brak *human-in-the-loop* i interrupts.

---

## 2. Rekomendowany stos technologiczny i struktura projektu

### 2.1 Stack technologiczny
* **Język:** Python 3.12+ (zarządzanie zależnościami przez `uv` lub `pip`).
* **Kontrakty i walidacja:** `pydantic>=2.0`
* **Klient HTTP:** `httpx` (asynchroniczny lub synchroniczny z timeoutami i retry).
* **Ekstrakcja PDF/HTML:** `pymupdf` (`fitz`) lub `pypdf` + `beautifulsoup4`.
* **Wyszukiwanie leksykalne (RAM):** `rank-bm25` + prosta normalizacja polskich słów (tokenizacja, usuwanie stopwords, opcjonalny stemmer/lematyzator, np. `stempel` / `morfeusz2` lub lekki stemmer regułowy bez ciężkich zależności binarnych).
* **Orkiestracja agentowa:** `langgraph>=1.0` (`StateGraph`, `ToolNode`, `tools_condition`) + `langchain-core>=1.0` (wiadomości, `@tool`, `Runnable`).
* **Klient LLM:** `langchain>=1.0` — `init_chat_model("provider:model")` jako jedyny punkt wyboru modelu (podmiana dostawcy wyłącznie przez `.env`, bez własnego adaptera). Integracje dostawców jako opcjonalne zależności: `langchain-anthropic`, `langchain-openai`, `langchain-google-genai`. Structured Outputs przez `model.with_structured_output(LlmEvaluationPayload, method="json_schema", include_raw=True)`.
* **Obserwowalność (opcjonalnie):** LangSmith — śledzenie przebiegu grafu włączane zmiennymi `LANGSMITH_TRACING` / `LANGSMITH_API_KEY`, bez zmian w kodzie.
* **Testy:** `pytest` + `pytest-asyncio` + `respx` (do mockowania zapytań HTTP) + fałszywe modele z `langchain_core` (`GenericFakeChatModel`) do deterministycznych testów grafu bez wywołań API.
* **CI/CD:** GitHub Actions (`.github/workflows/ci.yml`) — `uv sync --locked` + `uv run pytest` przy każdym pushu na `main` i w każdym pull requeście; bez anulowania trwających buildów, tak aby każdy commit miał własny przebieg. Testy `live` są w CI pomijane (brak klucza API).

### 2.2 Docelowa struktura katalogów
```text
legal-compliance-poc/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Pipeline CI (GitHub Actions): instalacja zależności + pytest
├── .env.example
├── pyproject.toml
├── README.md
├── src/
│   └── legal_checker/
│       ├── __init__.py
│       ├── config.py              # Konfiguracja (model LLM, limity top_k i wywołań narzędzi)
│       ├── models.py              # Kontrakty danych (Pydantic: Input, LegalUnit, ComplianceResult)
│       ├── eli_resolver.py        # Trójstopniowy resolver wskazówki -> klucz ELI
│       ├── eli_client.py          # Klient API ELI Sejmu + pobieranie metadanych i treści
│       ├── text_extractor.py      # Ekstrakcja tekstu z PDF/HTML i czyszczenie artefaktów
│       ├── legal_chunker.py       # Parser struktury prawnej (Dział/Rozdział/Art./Ust./Pkt)
│       ├── bm25_retriever.py      # Indeks BM25 w pamięci RAM + wyszukiwanie top-K kandydatów
│       ├── post_validator.py      # Deterministyczna walidacja wyniku LLM i dopięcie metadanych
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── state.py           # GraphState (TypedDict) + reducery
│       │   ├── llm.py             # Fabryka modeli: init_chat_model + bind_tools / with_structured_output
│       │   ├── prompts.py         # Prompty systemowe (research, ocena) po polsku
│       │   ├── tools.py           # Fabryka narzędzi read-only: search_provisions, get_provision
│       │   ├── nodes.py           # Węzły grafu (deterministyczne + LLM)
│       │   └── graph.py           # Budowa i kompilacja StateGraph, funkcja check()
│       └── cli.py                 # Interfejs CLI do uruchamiania i prezentacji wyników
└── tests/
    ├── fixtures/                  # Przykładowe fragmenty aktów i dokumentów
    ├── test_eli_resolver.py
    ├── test_legal_chunker.py
    ├── test_bm25_retriever.py
    ├── test_agent_tools.py
    ├── test_graph.py              # Przebiegi grafu na fałszywych modelach (routing, limity, retry)
    ├── test_post_validator.py
    └── test_e2e_cases.py          # 6 przypadków testowych ze specyfikacji zadania
```

---

## 3. Kontrakty danych (Data Schemas)

Modele domenowe powinny być zdefiniowane w `src/legal_checker/models.py` z użyciem **Pydantic v2**, a stan grafu w `src/legal_checker/agent/state.py`.

### 3.1 Wejście (`CheckRequest`)
```python
class CheckRequest(BaseModel):
    document_fragment: str = Field(..., min_length=10, description="Fragment dokumentu firmowego po polsku")
    act_hint: str = Field(..., min_length=2, description="Wskazówka aktu: ELI, link, sygnatura Dz.U./M.P. lub tytuł")
```

### 3.2 Jednostka redakcyjna po parsowaniu (`LegalUnit`)
```python
class LegalUnit(BaseModel):
    unit_id: str                     # np. "DU/2014/827:art.27"
    eli: str                         # np. "DU/2014/827"
    article: str                     # np. "27"
    paragraph: str | None = None     # opcjonalnie jeśli rozbijamy długie artykuły na ustępy
    hierarchy_path: list[str]        # np. ["Rozdział 4", "Art. 27"]
    text: str                        # Oczyszczony tekst jednostki redakcyjnej
```

### 3.3 Wyjście z LLM (`LlmEvaluationPayload`) vs. Wynik końcowy (`ComplianceResult`)
Model LLM zwraca **wyłącznie** pola wymagające oceny semantycznej (wymuszone przez JSON Schema), a kod dopina twarde metadane źródłowe:

```python
class ComplianceStatus(str, Enum):
    COMPLIANT = "ZGODNY"
    PARTIALLY_COMPLIANT = "CZĘŚCIOWO ZGODNY"
    NON_COMPLIANT = "NIEZGODNY"
    NOT_APPLICABLE = "NIE DOTYCZY / BRAK PODSTAW DO OCENY"

class LlmEvaluationPayload(BaseModel):
    status: ComplianceStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    document_excerpt: str | None = Field(
        None,
        description="Dosłowny cytat problematycznego (lub ocenianego) fragmentu z dokumentu wejściowego"
    )
    matched_provision: str | None = Field(
        None,
        description="Konkretny przepis z dostarczonego kontekstu, np. 'art. 27' lub 'art. 12 ust. 1 pkt 2'"
    )
    rationale: str = Field(..., description="Zwięzłe uzasadnienie oceny w 2-5 zdaniach po polsku")
    suggested_fix: str | None = Field(None, description="Opcjonalna propozycja poprawki brzmienia zapisu")

class LegalBasis(BaseModel):
    act_title: str
    eli: str
    provision: str | None
    as_of_date: str                  # ISO YYYY-MM-DD (data pobrania / aktualnego stanu)
    source_url: str

class ComplianceResult(BaseModel):
    status: ComplianceStatus
    confidence: float
    document_excerpt: str | None
    legal_basis: LegalBasis
    rationale: str
    suggested_fix: str | None
    consulted_units: list[str] = Field(default_factory=list)  # unit_id przepisów widzianych przez agenta
    warnings: list[str] = Field(default_factory=list)
```

### 3.4 Stan grafu (`GraphState`)
```python
class GraphState(TypedDict, total=False):
    request: CheckRequest
    eli_ref: EliRef
    act: ActDocument                                  # metadane (tytuł, source_url) + oczyszczony tekst
    units: list[LegalUnit]
    candidates: list[tuple[LegalUnit, float]]         # wynik BM25 (jednostka, score)
    messages: Annotated[list[AnyMessage], add_messages]  # konwersacja etapu research (AI + ToolMessage)
    llm_payload: LlmEvaluationPayload | None
    schema_errors: list[str]
    eval_attempts: int
    result: ComplianceResult
```
* Narzędzia zwracają `response_format="content_and_artifact"`: `content` to tekst przepisów dla LLM, a `artifact` to lista `unit_id`. Zbiór przepisów widzianych przez agenta (`consulted_units`) = `candidates` ∪ artefakty wszystkich `ToolMessage` w `messages` — wyliczany deterministycznie w post-validatorze, bez polegania na deklaracji LLM.

### 3.5 Topologia grafu
```text
START → resolve_act → fetch_act → chunk_act → retrieve
retrieve ──(max score BM25 == 0)──────────────→ short_circuit_na → END
retrieve ──────────────────────────────────────→ research
research ──(tool_calls i budżet niewyczerpany)─→ tools → research
research ──(brak tool_calls lub budżet wyczerpany)→ evaluate
evaluate ──(błąd schematu i eval_attempts < 2)─→ evaluate
evaluate ──────────────────────────────────────→ validate → END
```
* `research` i `evaluate` to **dwa osobne wywołania** LLM: `research` używa modelu z `bind_tools(...)`, `evaluate` — modelu z `with_structured_output(...)`. Rozdzielenie upraszcza prompty i omija ograniczenia dostawców w łączeniu wywołań narzędzi z wymuszonym JSON Schema.

---

## 4. Plan realizacji — Taski i Subtaski dla Claude Code

Realizuj zadania **etapowo**, po każdym tasku uruchamiając testy jednostkowe i weryfikując kryteria akceptacji (*Definition of Done*).

---

### ETAP 1: Fundament projektu i kontrakty danych

#### Task 1.1: Inicjalizacja szkieletu projektu i konfiguracji
* [ ] **Subtask 1.1.1:** Utwórz strukturę katalogów zgodną z sekcją 2.2 oraz plik `pyproject.toml` z zależnościami (`pydantic`, `httpx`, `pymupdf`, `beautifulsoup4`, `rank-bm25`, `langchain`, `langchain-core`, `langgraph`, `python-dotenv`, `rich`; opcjonalne grupy: `anthropic = ["langchain-anthropic"]`, `openai = ["langchain-openai"]`, `google = ["langchain-google-genai"]`; dev: `pytest`, `pytest-asyncio`, `respx`).
* [ ] **Subtask 1.1.2:** Utwórz `.env.example` i `src/legal_checker/config.py` (klasa ustawień ładująca m.in. `LLM_MODEL` w formacie `init_chat_model`, np. `"anthropic:claude-sonnet-5-5"`, klucz API dostawcy w jego standardowej zmiennej, np. `ANTHROPIC_API_KEY`, `ELI_API_BASE_URL="https://api.sejm.gov.pl/eli"`, `BM25_TOP_K=12`, `AGENT_MAX_TOOL_CALLS=4`, `GRAPH_RECURSION_LIMIT=20`, `CACHE_DIR=".cache"`, opcjonalnie `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`).
* [ ] **Subtask 1.1.3:** Zaimplementuj wszystkie modele Pydantic w `src/legal_checker/models.py` zgodnie z sekcjami 3.1–3.3 oraz `GraphState` w `src/legal_checker/agent/state.py` zgodnie z sekcją 3.4.
* [ ] **Subtask 1.1.4:** Dodaj pipeline CI w GitHub Actions (`.github/workflows/ci.yml`) zgodnie z sekcją 2.1: instalacja zależności przez `uv sync --locked` i uruchomienie `pytest` przy pushu na `main` oraz w pull requestach.
* **DoD:** Import `legal_checker.models`, `legal_checker.config` i `legal_checker.agent.state` działa bez błędów, a schemat JSON generuje się poprawnie przez `LlmEvaluationPayload.model_json_schema()`.

---

### ETAP 2: Integracja z API ELI i pozyskiwanie treści aktu

#### Task 2.1: Trójstopniowy resolver wskazówki aktu (`eli_resolver.py`)
* [ ] **Subtask 2.1.1:** Zaimplementuj normalizację wejścia `act_hint` (trim, ujednolicenie białych znaków i myślników).
* [ ] **Subtask 2.1.2:** **Ścieżka 1 (Bezpośredni ELI / URL):** Wykrywanie wzorca `(DU|MP)/(\d{4})/(\d+)` w surowym stringu lub w linkach (np. `eli.gov.pl/...`, `api.sejm.gov.pl/eli/...`).
* [ ] **Subtask 2.1.3:** **Ścieżka 2 (Sygnatura publikacyjna):** Parser regex obsługujący zapisy typu `Dz.U. z 2014 r. poz. 827`, `Dz. U. 2014 poz. 827`, `Dz.U. 2014, nr 12, poz. 827`, `M.P. 2023 poz. 100` → mapowanie na `{publisher}/{year}/{position}`.
* [ ] **Subtask 2.1.4:** **Ścieżka 3 (Wyszukiwanie po tytule):** Jeśli ścieżki 1–2 nie zadziałają, wywołanie wyszukiwarki API ELI (`/acts/search?title=...`) i wybór najbardziej trafnego kandydata (preferując ustawy/rozporządzenia o statusie obowiązującym). Zwrócenie czytelnego wyjątku `ActResolutionError`, jeśli brak dopasowania.
* [ ] **Subtask 2.1.5:** Napisz testy jednostkowe w `tests/test_eli_resolver.py` pokrywające wszystkie 3 ścieżki oraz przypadki błędne.
* **DoD:** Wszystkie warianty zapisu sygnatury i linków mapują się poprawnie na `EliRef(publisher, year, position)`, a wyszukiwanie po tytule zwraca poprawny klucz ELI.

#### Task 2.2: Klient API ELI i wybór aktualnego tekstu (`eli_client.py`)
* [ ] **Subtask 2.2.1:** Zaimplementuj pobieranie metadanych aktu z `GET /acts/{publisher}/{year}/{position}` (tytuł aktu, status, daty, informacje o tekstach/powiązaniach).
* [ ] **Subtask 2.2.2:** Dodaj logikę sprawdzania, czy dla danego aktu dostępny jest nowszy tekst jednolity / ujednolicony (w metadanych lub powiązaniach `/references`), aby w PoC pracować na najbardziej aktualnym brzmieniu „na dzień dzisiejszy”.
* [ ] **Subtask 2.2.3:** Zaimplementuj pobieranie treści dokumentu (preferuj HTML, jeśli dostępny; w przeciwnym razie pobierz `text.pdf`).
* [ ] **Subtask 2.2.4:** Dodaj prosty lokalny cache (w pamięci RAM + zapis plików surowych/sparsowanych w `.cache/`), aby nie pobierać tego samego aktu wielokrotnie podczas testów.
* **DoD:** Klient potrafi pobrać metadane i plik treści np. dla `DU/2014/827` (ustawa o prawach konsumenta) i zapisać je w cache.

#### Task 2.3: Ekstrakcja i czyszczenie tekstu (`text_extractor.py`)
* [ ] **Subtask 2.3.1:** Zaimplementuj ekstrakcję tekstu z PDF (za pomocą `pymupdf` z zachowaniem kolejności bloków tekstu) oraz z HTML (`BeautifulSoup`).
* [ ] **Subtask 2.3.2:** Zaimplementuj czyszczenie artefaktów dzienników urzędowych: usuwanie powtarzających się nagłówków/stopek stron (np. `©Kancelaria Sejmu`, numery stron `s. 1/45`, daty generowania pliku).
* [ ] **Subtask 2.3.3:** Zaimplementuj łączenie wyrazów podzielonych myślnikiem na końcu linii (`([a-ząćęłńóśźż])-\n([a-ząćęłńóśźż])` → `\1\2`) przy jednoczesnym zachowaniu podziałów akapitów.
* **DoD:** Wyekstrahowany tekst ustawy jest ciągły, pozbawiony nagłówków stron i połamanych wyrazów, gotowy do cięcia strukturalnego.

---

### ETAP 3: Cięcie po strukturze prawnej i indeksowanie BM25 w RAM

#### Task 3.1: Parser struktury prawnej (`legal_chunker.py`)
* [ ] **Subtask 3.1.1:** Zaimplementuj maszynę stanów / parser liniowy wykrywający nagłówki nadrzędne (`Księga`, `Tytuł`, `Dział`, `Rozdział`, `Oddział`) i utrzymujący aktualny stos `hierarchy_path`.
* [ ] **Subtask 3.1.2:** Zaimplementuj wykrywanie granic artykułów (`^Art\.\s*(\d+[a-z]?)\.`) oraz opcjonalnie paragrafów dla rozporządzeń (`^§\s*(\d+[a-z]?)\.`).
* [ ] **Subtask 3.1.3:** Zadbaj o dzielenie bardzo długich artykułów (np. powyżej 3000 znaków) na mniejsze jednostki po ustępach (`^(\d+[a-z]?)\.`) z zachowaniem kontekstu artykułu, a standardowe artykuły trzymaj w całości z wyraźną numeracją ustępów i punktów.
* [ ] **Subtask 3.1.4:** Dodaj flagę ostrzegawczą, jeśli dokument nie zawiera rozpoznawalnych jednostek `Art.` / `§` (fallback na podział akapitowy).
* [ ] **Subtask 3.1.5:** Napisz testy w `tests/test_legal_chunker.py` na przykładowym fragmencie ustawy wielorozdziałowej.
* **DoD:** Parser generuje listę obiektów `LegalUnit` z poprawnym `article`, `hierarchy_path` i pełną treścią przepisu.

#### Task 3.2: Retriever BM25 w pamięci operacyjnej (`bm25_retriever.py`)
* [ ] **Subtask 3.2.1:** Zaimplementuj lekki tokenizer dla języka polskiego (lowercase, usunięcie interpunkcji, filtrowanie krótkich stopwords typu *oraz, lub, w, z, na, do, o, że, się, jest*, prosta normalizacja końcówek fleksyjnych / stemmer).
* [ ] **Subtask 3.2.2:** Zbuduj klasę `InMemoryBm25Index`, która przyjmuje `list[LegalUnit]`, tworzy korpus tokenów w RAM i inicjalizuje `BM25Okapi` (z `rank_bm25`).
* [ ] **Subtask 3.2.3:** Zaimplementuj metodę `search(query: str, top_k: int = 12) -> list[tuple[LegalUnit, float]]`, metodę `get(article: str, paragraph: str | None = None) -> LegalUnit | None` (dokładne pobranie jednostki po numerze — używane przez narzędzie agenta) oraz prostą heurystykę dociągania przywołanych artykułów (jeśli w top wynikach pojawia się np. odwołanie `"o którym mowa w art. 27"`, dołącz również `art. 27`, o ile nie ma go w top-K). Heurystyka zapewnia bazowy kontekst; agent może go uzupełnić w Etapie 4.
* [ ] **Subtask 3.2.4:** Napisz testy w `tests/test_bm25_retriever.py` weryfikujące trafność wyszukiwania dla przykładowych klauzul umownych.
* **DoD:** Wyszukiwanie dla fragmentu o odstąpieniu od umowy zwraca właściwe artykuły ustawy w top-5 wyników w czasie < 50 ms w pamięci RAM.

---

### ETAP 4: Warstwa agentowa (LangChain / LangGraph), ocena i walidacja

#### Task 4.1: Narzędzia agenta (`agent/tools.py`)
* [ ] **Subtask 4.1.1:** Zaimplementuj fabrykę `make_tools(index: InMemoryBm25Index) -> list[BaseTool]`, która tworzy narzędzia domknięte (*closure*) na indeksie bieżącego żądania (bez stanu globalnego), dekorowane `@tool(response_format="content_and_artifact")` z polskimi opisami i typowanymi argumentami:
  * `search_provisions(query: str, top_k: int = 5)` — dodatkowe zapytanie BM25, np. o pojęcie użyte w klauzuli, którego nie objęło pierwsze wyszukiwanie.
  * `get_provision(article: str, paragraph: str | None = None)` — dokładne pobranie przepisu po numerze, np. artykułu przywołanego w treści kandydata.
* [ ] **Subtask 4.1.2:** Każde narzędzie zwraca `(content, artifact)`: `content` to sformatowany tekst przepisów z nagłówkiem `[unit_id] hierarchy_path`, przycięty do rozsądnego limitu znaków; `artifact` to lista zwróconych `unit_id`. Brak wyników → czytelny komunikat w `content` i pusta lista (bez wyjątku).
* [ ] **Subtask 4.1.3:** Napisz testy w `tests/test_agent_tools.py` (wywołanie narzędzi przez `.invoke(...)` z `ToolCall`, sprawdzenie `content` i `artifact`, przypadek nieistniejącego artykułu).
* **DoD:** Narzędzia działają deterministycznie, bez sieci, a ich schematy argumentów (`tool.args_schema`) generują się poprawnie.

#### Task 4.2: Fabryka modeli i prompty (`agent/llm.py`, `agent/prompts.py`)
* [ ] **Subtask 4.2.1:** Zaimplementuj `build_models(settings) -> tuple[Runnable, Runnable]` zwracającą: (a) model research = `init_chat_model(LLM_MODEL, temperature=0.0).bind_tools(tools)`, (b) model oceny = `init_chat_model(...).with_structured_output(LlmEvaluationPayload, method="json_schema", include_raw=True)`. Modele muszą być **wstrzykiwalne** do budowy grafu (testy podstawiają fałszywe modele).
* [ ] **Subtask 4.2.2:** Zaprojektuj prompt systemowy etapu **research** po polsku: przejrzyj kandydatów z `<legal_provisions>`, a jeśli brakuje przepisu przywołanego w ich treści lub kluczowego dla fragmentu dokumentu — użyj narzędzi; nie szukaj „na zapas”; gdy kontekst jest wystarczający, zakończ bez wywołania narzędzi.
* [ ] **Subtask 4.2.3:** Zaprojektuj prompt systemowy etapu **oceny** po polsku z twardymi regułami:
  * Oceniaj **wyłącznie** na podstawie przepisów z sekcji `<legal_provisions>` oraz przepisów zwróconych przez narzędzia.
  * Nie dopowiadaj przepisów spoza kontekstu.
  * Jeśli żaden z dostarczonych przepisów nie odnosi się merytorycznie do fragmentu dokumentu, wybierz kategorię `NIE DOTYCZY / BRAK PODSTAW DO OCENY`.
  * W polu `document_excerpt` podaj **dosłowny** wycinek tekstu z `<document_fragment>`.
  * Uzasadnienie (`rationale`) ma mieć **2–5 zwięzłych zdań**.
* **DoD:** `build_models` zwraca działające modele dla skonfigurowanego dostawcy, a podmiana `LLM_MODEL` w `.env` (np. z `anthropic:...` na `openai:...`) nie wymaga zmian w kodzie.

#### Task 4.3: Węzły i graf (`agent/nodes.py`, `agent/graph.py`)
* [ ] **Subtask 4.3.1:** Zaimplementuj węzły deterministyczne jako funkcje `(state: GraphState) -> dict` (zwracające częściową aktualizację stanu): `resolve_act`, `fetch_act`, `chunk_act`, `retrieve`, `short_circuit_na` (gotowy `ComplianceResult` ze statusem `NIE DOTYCZY / BRAK PODSTAW DO OCENY` bez wywołania LLM) oraz `validate` (wywołuje post-validator z Task 4.4).
* [ ] **Subtask 4.3.2:** Zaimplementuj węzeł `research`: przy pierwszym wejściu buduje wiadomości (prompt systemowy + `<document_fragment>` + `<legal_provisions>` z kandydatami), wywołuje model research i dopisuje `AIMessage` do `messages`.
* [ ] **Subtask 4.3.3:** Zaimplementuj router po `research`: przejście do `ToolNode(tools)` tylko, jeśli ostatnia wiadomość zawiera `tool_calls` **i** liczba dotychczasowych `ToolMessage` < `AGENT_MAX_TOOL_CALLS`; w przeciwnym razie przejście do `evaluate` (po wyczerpaniu budżetu dodaj ostrzeżenie `"Osiągnięto limit wywołań narzędzi agenta"`). Nadmiarowe `tool_calls` z ostatniej `AIMessage` nie są wykonywane.
* [ ] **Subtask 4.3.4:** Zaimplementuj węzeł `evaluate`: przekazuje modelowi oceny fragment, kandydatów i treść wszystkich `ToolMessage`; przy `parsing_error` z `include_raw=True` zapisuje błąd w `schema_errors` i zwiększa `eval_attempts`. Krawędź warunkowa ponawia `evaluate` maksymalnie 1 raz (z dołączonym komunikatem błędu); po drugim niepowodzeniu graf kończy się czytelnym wyjątkiem `EvaluationError`. Błędy API dostawcy obsługuj przez `.with_retry(stop_after_attempt=2)` na modelu.
* [ ] **Subtask 4.3.5:** W `graph.py` zbuduj `build_graph(research_model, eval_model, tools) -> CompiledStateGraph` zgodnie z topologią z sekcji 3.5 oraz funkcję `check(request: CheckRequest) -> ComplianceResult`, która tworzy indeks, narzędzia i modele dla żądania i wywołuje graf z `config={"recursion_limit": GRAPH_RECURSION_LIMIT}`.
* [ ] **Subtask 4.3.6:** Napisz testy w `tests/test_graph.py` na fałszywych modelach (`GenericFakeChatModel` dla research zwracający zaprogramowane `AIMessage` z `tool_calls`, `RunnableLambda` zwracający `{"parsed": LlmEvaluationPayload(...), "parsing_error": None, "raw": ...}` dla oceny) i zamockowanym ELI (`respx` / fixture z cache):
  * krótkie spięcie przy BM25 = 0 — model nie jest wywoływany,
  * ścieżka bez narzędzi (research od razu kończy),
  * ścieżka z `get_provision` — dociągnięty przepis trafia do `consulted_units`,
  * limit `AGENT_MAX_TOOL_CALLS` — pętla zostaje przerwana i pojawia się ostrzeżenie,
  * pojedynczy retry po błędzie schematu oraz `EvaluationError` po dwóch błędach.
* **DoD:** `graph.get_graph().draw_mermaid()` odzwierciedla topologię z sekcji 3.5, a wszystkie testy grafu przechodzą bez dostępu do sieci i bez klucza API.

#### Task 4.4: Deterministyczny post-validator (`post_validator.py`)
* [ ] **Subtask 4.4.1:** Zweryfikuj, czy `document_excerpt` (jeśli nie jest `None`) faktycznie występuje jako podciąg w oryginalnym `document_fragment` (po normalizacji białych znaków). Jeśli model sparafrazował cytat, dodaj ostrzeżenie do `warnings` i obniż `confidence`.
* [ ] **Subtask 4.4.2:** Wylicz `consulted_units` = `unit_id` kandydatów BM25 ∪ artefakty wszystkich `ToolMessage` ze stanu. Zweryfikuj, czy wskazany `matched_provision` (numer artykułu) należy do `consulted_units`. Jeśli nie — oznacz ostrzeżenie w `warnings` (przepis spoza kontekstu widzianego przez agenta).
* [ ] **Subtask 4.4.3:** Złóż finalny obiekt `ComplianceResult`, wstrzykując deterministycznie z kodu `act_title`, `eli`, `source_url`, `consulted_units` oraz `as_of_date = date.today().isoformat()`.
* [ ] **Subtask 4.4.4:** Napisz testy jednostkowe w `tests/test_post_validator.py` (m.in. dla wykrywania zmyślonego cytatu oraz przepisu spoza `consulted_units`).
* **DoD:** Post-validator deterministycznie dokleja metadane źródłowe i wyłapuje rozbieżności w cytatach i przepisach bez udziału LLM.

---

### ETAP 5: CLI i zestaw testów End-to-End

#### Task 5.1: Interfejs CLI (`cli.py`)
* [ ] **Subtask 5.1.1:** Zbuduj czytelne CLI w `cli.py` (używając `argparse` lub `typer` + `rich`), które pozwala przekazać `--text` (lub `--file`) oraz `--hint`, wywołuje `agent.graph.check(...)`, a następnie wypisuje zarówno surowy **JSON**, jak i czytelne podsumowanie dla człowieka (w tym listę `consulted_units` i wywołań narzędzi agenta).
* [ ] **Subtask 5.1.2:** Dodaj flagę `--show-graph`, która wypisuje diagram grafu w formacie Mermaid (`graph.get_graph().draw_mermaid()`), oraz flagę `--verbose`, która strumieniuje kolejne kroki grafu (`graph.stream(..., stream_mode="updates")`).
* **DoD:** Uruchomienie komendy z terminala zwraca pełny wynik JSON oraz czytelny raport w konsoli; `--show-graph` wypisuje topologię zgodną z sekcją 3.5.

#### Task 5.2: Implementacja 6 przypadków testowych (`tests/test_e2e_cases.py`)
* [ ] **Subtask 5.2.1:** Przygotuj zestaw 6 przypadków testowych pokrywających wszystkie kategorie oceny i przypadki graniczne:
  1. **ZGODNY:** Poprawny zapis o 14-dniowym prawie odstąpienia od umowy zawartej na odległość (`DU/2014/827`).
  2. **CZĘŚCIOWO ZGODNY:** Klauzula informacyjna RODO / konsumencka zawierająca część obowiązkowych informacji, ale pomijająca kluczowy element (np. termin lub sposób złożenia oświadczenia).
  3. **NIEZGODNY:** Zapis skracający ustawowy termin odstąpienia od umowy konsumenckiej do 7 dni lub wyłączający ustawowe uprawnienie konsumenta.
  4. **NIE DOTYCZY / BRAK PODSTAW DO OCENY:** Zapis czysto organizacyjny (np. godziny spotkań wewnętrznego zespołu) sprawdzany względem ustawy o prawach konsumenta.
  5. **Przypadek graniczny / niejednoznaczny:** Zapis o karze umownej / odstąpieniu bez jasnego rozstrzygnięcia, czy dotyczy relacji B2B czy B2C — weryfikacja niższej pewności (`confidence`) i ostrożnego uzasadnienia.
  6. **Rozwiązywanie różnych formatów wskazówki:** Ten sam przypadek uruchomiony dla pełnego linku ELI, sygnatury `Dz.U. z 2014 r. poz. 827` oraz nazwy `"ustawa o prawach konsumenta"`.
* [ ] **Subtask 5.2.2:** Testy e2e z prawdziwym LLM oznacz markerem `@pytest.mark.live` (pomijane bez klucza API); w trybie domyślnym uruchamiaj je na fałszywych modelach i akcie z cache, weryfikując przebieg grafu, a nie treść oceny.
* **DoD:** Wszystkie 6 scenariuszy przechodzi w trybie z zamockowanym/cache'owanym aktem i fałszywymi modelami oraz działa poprawnie w trybie live z API ELI i prawdziwym LLM.

---

## 5. Wskazówki do pracy z Claude Code

1. **Pracuj etapami:** Wklej ten plik do repozytorium jako `SPEC.md` i proś Claude Code o realizację **jednego taska naraz** (np. *"Zrealizuj Task 2.1 z SPEC.md wraz z testami jednostkowymi i uruchom pytest"*).
2. **Weryfikuj ekstrakcję z API ELI na żywo:** Po ukończeniu Etapu 2 wykonaj jeden prawdziwy strzał do `api.sejm.gov.pl/eli/acts/DU/2014/827`, aby upewnić się, że format odpowiedzi i czyszczenie PDF/HTML zgadza się z rzeczywistym kształtem danych z Sejmu.
3. **Najpierw graf na fałszywych modelach:** W Etapie 4 zbuduj i przetestuj topologię grafu (Task 4.3) na fałszywych modelach, zanim podłączysz prawdziwego dostawcę LLM — routing, limity i retry muszą działać deterministycznie.
4. **Śledź przebieg agenta:** Przy strojeniu promptów włącz LangSmith (`LANGSMITH_TRACING=true`) lub użyj `--verbose`, aby zobaczyć, które narzędzia agent wywołuje i czy nie szuka „na zapas”.
5. **Zapisz przykładowy wynik JSON:** Wynik z Taska 5.2 (np. dla przypadku `CZĘŚCIOWO ZGODNY` lub `NIEZGODNY`) zapisz jako przykład do pokazania podczas rozmowy rekrutacyjnej, razem z diagramem grafu z `--show-graph`.
