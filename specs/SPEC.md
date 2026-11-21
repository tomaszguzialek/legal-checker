# Specyfikacja Implementacyjna (PoC): AI Legal & Compliance Checker

**Wersja:** 1.0 (PoC)  
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
5. Ocenia zgodność fragmentu z wybranymi przepisami za pomocą **LLM z wymuszonym Structured Outputs (JSON Schema)** oraz przeprowadza deterministyczną walidację wyniku w kodzie.

### 1.2 Podział odpowiedzialności: Kod deterministyczny vs. LLM
* **Kod deterministyczny (Python):**
  * Normalizacja i trójstopniowe parsowanie wskazówki aktu na klucz ELI.
  * Komunikacja z API ELI, wybór najbardziej aktualnego tekstu (np. tekst ujednolicony/jednolity), ekstrakcja tekstu z PDF/HTML i czyszczenie nagłówków/stopek.
  * Parsowanie struktury prawnej (budowa drzewa jednostek redakcyjnych).
  * Indeksowanie i wyszukiwanie kandydatów w pamięci za pomocą **BM25** + prosty cache w RAM / na dysku lokalnym.
  * Wstrzykiwanie twardych metadanych źródłowych (`source_url`, `as_of_date`, `eli`, `act_title`) oraz walidacja post-LLM (sprawdzenie, czy cytat pochodzi z wejścia, walidacja schematu).
* **Model językowy (LLM):**
  * Wyłącznie merytoryczna ocena zgodności na podstawie dostarczonego tekstu przepisów i fragmentu dokumentu, wskazanie konkretnej jednostki redakcyjnej (`art.`, `ust.`, `pkt`), zacytowanie problematycznego fragmentu dokumentu oraz wygenerowanie zwięzłego uzasadnienia (2–5 zdań) i opcjonalnej propozycji poprawki.

### 1.3 Świadome uproszczenia w PoC (Out of Scope)
* Brak obsługi dat wstecznych (`as_of_date` ustawiane automatycznie na bieżącą datę pobrania).
* Brak automatycznego zgadywania aktu prawnego bez żadnej wskazówki (*wariant ambitny* odłożony na etap produkcyjny).
* Brak bazy danych (PostgreSQL), bazy wektorowej (`pgvector`/Qdrant) i embeddingów — wszystko działa w pamięci operacyjnej procesu (z opcjonalnym lokalnym cache plików `.json` w katalogu `.cache/`).

---

## 2. Rekomendowany stos technologiczny i struktura projektu

### 2.1 Stack technologiczny
* **Język:** Python 3.12+ (zarządzanie zależnościami przez `uv` lub `pip`).
* **Kontrakty i walidacja:** `pydantic>=2.0`
* **Klient HTTP:** `httpx` (asynchroniczny lub synchroniczny z timeoutami i retry).
* **Ekstrakcja PDF/HTML:** `pymupdf` (`fitz`) lub `pypdf` + `beautifulsoup4`.
* **Wyszukiwanie leksykalne (RAM):** `rank-bm25` + prosta normalizacja polskich słów (tokenizacja, usuwanie stopwords, opcjonalny stemmer/lematyzator, np. `stempel` / `morfeusz2` lub lekki stemmer regułowy bez ciężkich zależności binarnych).
* **Klient LLM:** SDK z natywnym wsparciem dla **Structured Outputs (JSON Schema)** (np. `openai` / `anthropic` / `google-genai` — skonfigurowane przez interfejs adaptera, aby łatwo podmieniać model w `.env`).
* **Testy:** `pytest` + `pytest-asyncio` + `respx` (do mockowania zapytań HTTP w testach jednostkowych).

### 2.2 Docelowa struktura katalogów
```text
legal-compliance-poc/
├── .env.example
├── pyproject.toml
├── README.md
├── src/
│   └── legal_checker/
│       ├── __init__.py
│       ├── config.py              # Konfiguracja (klucze API, model LLM, limity top_k)
│       ├── models.py              # Kontrakty danych (Pydantic: Input, LegalUnit, ComplianceResult)
│       ├── eli_resolver.py        # Trójstopniowy resolver wskazówki -> klucz ELI
│       ├── eli_client.py          # Klient API ELI Sejmu + pobieranie metadanych i treści
│       ├── text_extractor.py      # Ekstrakcja tekstu z PDF/HTML i czyszczenie artefaktów
│       ├── legal_chunker.py       # Parser struktury prawnej (Dział/Rozdział/Art./Ust./Pkt)
│       ├── bm25_retriever.py      # Indeks BM25 w pamięci RAM + wyszukiwanie top-K kandydatów
│       ├── evaluator.py           # Prompt + wywołanie LLM ze Structured Outputs
│       ├── post_validator.py      # Deterministyczna walidacja wyniku LLM i dopięcie metadanych
│       ├── pipeline.py            # Orkiestrator całego przepływu end-to-end
│       └── cli.py                 # Interfejs CLI do uruchamiania i prezentacji wyników
└── tests/
    ├── fixtures/                  # Przykładowe fragmenty aktów i dokumentów
    ├── test_eli_resolver.py
    ├── test_legal_chunker.py
    ├── test_bm25_retriever.py
    ├── test_post_validator.py
    └── test_e2e_cases.py          # 6 przypadków testowych ze specyfikacji zadania
```

---

## 3. Kontrakty danych (Data Schemas)

Wszystkie modele powinny być zdefiniowane w `src/legal_checker/models.py` z użyciem **Pydantic v2**.

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
    warnings: list[str] = Field(default_factory=list)
```

---

## 4. Plan realizacji — Taski i Subtaski dla Claude Code

Realizuj zadania **etapowo**, po każdym tasku uruchamiając testy jednostkowe i weryfikując kryteria akceptacji (*Definition of Done*).

---

### ETAP 1: Fundament projektu i kontrakty danych

#### Task 1.1: Inicjalizacja szkieletu projektu i konfiguracji
* [ ] **Subtask 1.1.1:** Utwórz strukturę katalogów zgodną z sekcją 2.2 oraz plik `pyproject.toml` z zależnościami (`pydantic`, `httpx`, `pymupdf`, `beautifulsoup4`, `rank-bm25`, `python-dotenv`, `rich` oraz dev: `pytest`, `respx`).
* [ ] **Subtask 1.1.2:** Utwórz `.env.example` i `src/legal_checker/config.py` (klasa ustawień ładująca m.in. `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `ELI_API_BASE_URL="https://api.sejm.gov.pl/eli"`, `BM25_TOP_K=12`, `CACHE_DIR=".cache"`).
* [ ] **Subtask 1.1.3:** Zaimplementuj wszystkie modele Pydantic w `src/legal_checker/models.py` zgodnie z sekcją 3.
* **DoD:** Import `legal_checker.models` i `legal_checker.config` działa bez błędów, a schemat JSON generuje się poprawnie przez `LlmEvaluationPayload.model_json_schema()`.

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
* [ ] **Subtask 3.2.3:** Zaimplementuj metodę `search(query: str, top_k: int = 12) -> list[tuple[LegalUnit, float]]` oraz prostą heurystykę dociągania przywołanych artykułów (jeśli w top wynikach pojawia się np. odwołanie `"o którym mowa w art. 27"`, dołącz również `art. 27`, o ile nie ma go w top-K).
* [ ] **Subtask 3.2.4:** Napisz testy w `tests/test_bm25_retriever.py` weryfikujące trafność wyszukiwania dla przykładowych klauzul umownych.
* **DoD:** Wyszukiwanie dla fragmentu o odstąpieniu od umowy zwraca właściwe artykuły ustawy w top-5 wyników w czasie < 50 ms w pamięci RAM.

---

### ETAP 4: Ocena zgodności przez LLM (Structured Outputs) i walidacja

#### Task 4.1: Moduł oceny zgodności (`evaluator.py`)
* [ ] **Subtask 4.1.1:** Zaprojektuj prompt systemowy i użytkownika w języku polskim z twardymi regułami:
  * Oceniaj **wyłącznie** na podstawie dostarczonych przepisów z sekcji `<legal_provisions>`.
  * Nie dopowiadaj przepisów spoza kontekstu.
  * Jeśli żaden z dostarczonych przepisów nie odnosi się merytorycznie do fragmentu dokumentu, wybierz kategorię `NIE DOTYCZY / BRAK PODSTAW DO OCENY`.
  * W polu `document_excerpt` podaj **dosłowny** wycinek tekstu z `<document_fragment>`.
  * Uzasadnienie (`rationale`) ma mieć **2–5 zwięzłych zdań**.
* [ ] **Subtask 4.1.2:** Skonfiguruj wywołanie API LLM z wymuszeniem schematu JSON (`LlmEvaluationPayload`) oraz niską temperaturą (`temperature=0.0`).
* [ ] **Subtask 4.1.3:** Dodaj obsługę błędów API i maksymalnie 1 automatyczne ponowienie (*retry*), jeśli walidacja schematu zwróci błąd.
* **DoD:** Wywołanie `evaluate_compliance(fragment, candidate_units)` zwraca poprawnie zwalidowany obiekt `LlmEvaluationPayload`.

#### Task 4.2: Deterministyczny post-validator (`post_validator.py`)
* [ ] **Subtask 4.2.1:** Zweryfikuj, czy `document_excerpt` (jeśli nie jest `None`) faktycznie występuje jako podciąg w oryginalnym `document_fragment` (po normalizacji białych znaków). Jeśli model sparafrazował cytat, dodaj ostrzeżenie do `warnings` i obniż `confidence`.
* [ ] **Subtask 4.2.2:** Zweryfikuj, czy wskazany `matched_provision` (numer artykułu) znajduje się wśród przekazanych kandydatów `LegalUnit`. Jeśli nie — oznacz ostrzeżenie w `warnings`.
* [ ] **Subtask 4.2.3:** Złóż finalny obiekt `ComplianceResult`, wstrzykując deterministycznie z kodu `act_title`, `eli`, `source_url` oraz `as_of_date = date.today().isoformat()`.
* [ ] **Subtask 4.2.4:** Napisz testy jednostkowe w `tests/test_post_validator.py` (m.in. dla wykrywania zmyślonego cytatu).
* **DoD:** Post-validator deterministycznie dokleja metadane źródłowe i wyłapuje rozbieżności w cytatach bez udziału LLM.

---

### ETAP 5: Orkiestracja, CLI i zestaw testów End-to-End

#### Task 5.1: Orkiestrator potoku i CLI (`pipeline.py`, `cli.py`)
* [ ] **Subtask 5.1.1:** Połącz wszystkie kroki w klasie/funkcji `CompliancePipeline.check(request: CheckRequest) -> ComplianceResult`:
  1. `resolve_act_hint` → `EliRef`
  2. `EliClient.fetch_act` → tekst + metadane (z cache)
  3. `LegalChunker.parse` → `list[LegalUnit]`
  4. `InMemoryBm25Index.search` → `top_k` kandydatów (jeśli wszystkie wyniki BM25 mają `score == 0`, zwróć od razu `NIE DOTYCZY / BRAK PODSTAW DO OCENY` bez kosztownego wywołania LLM!)
  5. `Evaluator.evaluate` → `LlmEvaluationPayload`
  6. `PostValidator.finalize` → `ComplianceResult`
* [ ] **Subtask 5.1.2:** Zbuduj czytelne CLI w `cli.py` (używając `argparse` lub `typer` + `rich`), które pozwala przekazać `--text` (lub `--file`) oraz `--hint`, a następnie wypisuje zarówno surowy **JSON**, jak i czytelne podsumowanie dla człowieka.
* **DoD:** Uruchomienie komendy z terminala zwraca pełny wynik JSON oraz czytelny raport w konsoli.

#### Task 5.2: Implementacja 6 przypadków testowych (`tests/test_e2e_cases.py`)
* [ ] **Subtask 5.2.1:** Przygotuj zestaw 6 przypadków testowych pokrywających wszystkie kategorie oceny i przypadki graniczne:
  1. **ZGODNY:** Poprawny zapis o 14-dniowym prawie odstąpienia od umowy zawartej na odległość (`DU/2014/827`).
  2. **CZĘŚCIOWO ZGODNY:** Klauzula informacyjna RODO / konsumencka zawierająca część obowiązkowych informacji, ale pomijająca kluczowy element (np. termin lub sposób złożenia oświadczenia).
  3. **NIEZGODNY:** Zapis skracający ustawowy termin odstąpienia od umowy konsumenckiej do 7 dni lub wyłączający ustawowe uprawnienie konsumenta.
  4. **NIE DOTYCZY / BRAK PODSTAW DO OCENY:** Zapis czysto organizacyjny (np. godziny spotkań wewnętrznego zespołu) sprawdzany względem ustawy o prawach konsumenta.
  5. **Przypadek graniczny / niejednoznaczny:** Zapis o karze umownej / odstąpieniu bez jasnego rozstrzygnięcia, czy dotyczy relacji B2B czy B2C — weryfikacja niższej pewności (`confidence`) i ostrożnego uzasadnienia.
  6. **Rozwiązywanie różnych formatów wskazówki:** Ten sam przypadek uruchomiony dla pełnego linku ELI, sygnatury `Dz.U. z 2014 r. poz. 827` oraz nazwy `"ustawa o prawach konsumenta"`.
* **DoD:** Wszystkie 6 scenariuszy przechodzi w trybie z zamockowanym/cache'owanym aktem oraz działa poprawnie w trybie live z API ELI.

---

## 5. Wskazówki do pracy z Claude Code

1. **Pracuj etapami:** Wklej ten plik do repozytorium jako `SPEC.md` i proś Claude Code o realizację **jednego taska naraz** (np. *"Zrealizuj Task 2.1 z SPEC.md wraz z testami jednostkowymi i uruchom pytest"*).
2. **Weryfikuj ekstrakcję z API ELI na żywo:** Po ukończeniu Etapu 2 wykonaj jeden prawdziwy strzał do `api.sejm.gov.pl/eli/acts/DU/2014/827`, aby upewnić się, że format odpowiedzi i czyszczenie PDF/HTML zgadza się z rzeczywistym kształtem danych z Sejmu.
3. **Zapisz przykładowy wynik JSON:** Wynik z Taska 5.2 (np. dla przypadku `CZĘŚCIOWO ZGODNY` lub `NIEZGODNY`) zapisz jako przykład do pokazania podczas rozmowy rekrutacyjnej.