# ADR 0002: Unified text PDF over HTML as the source of an act's current wording

**Status:** Accepted  
**Date:** 2026-11-21  
**Applies to:** `specs/SPEC.md` (version 1.1), section 1.3 ("current legal state only"), Subtasks 2.2.2–2.2.3 and 2.3.2

---

## Context

The PoC assesses documents against the law as it stands today. SPEC originally said to fetch an act's content as HTML when available, and as `text.pdf` otherwise.

The act metadata from the Sejm ELI API (`GET /acts/{publisher}/{year}/{position}`) lists several text variants in `texts[]`. Checked against the live API for `DU/2014/827` (ustawa o prawach konsumenta) on 2026-11-21:

| `type` | File | Content |
|---|---|---|
| `H` | `text.html` | Wording as originally published (2014) |
| `O` / `I` | `D20140827.pdf` | The Dziennik Ustaw issue as published |
| `T` | `D20140827L.pdf` | Kancelaria Sejmu copy of the original wording |
| `U` | `D20140827Lj.pdf` | **Unified text** ("tekst ujednolicony"): the current wording, with all amendments applied, dated with the act's last change |

`text.html` has no `Art. 7aa`, which the 2022 amendment (`DU/2022/2337`) added. The `U` PDF has it. The same pattern holds for other amended acts, e.g. Kodeks cywilny (`DU/1964/93`) and the 2018 data protection act (`DU/2018/1000`).

Acts with official consolidated texts ("tekst jednolity") also list them under the relation `Inf. o tekście jednolitym`. Each is a separate act (an `Obwieszczenie` of the Marshal of the Sejm) with its own PDFs. It only covers amendments up to its own date.

## Decision

Pick the text source in this order:

1. the act's own unified text PDF (`U`),
2. the newest official consolidated text from `Inf. o tekście jednolitym`, using its `U`, `T` or `O` PDF,
3. `text.html`,
4. `text.pdf`.

HTML is used only when neither the unified text nor a consolidated text exists, i.e. for acts that were never amended. For those, the original wording is the current one.

## Rationale

* **Correctness first:** checking a document against wording from years ago would produce confident but wrong assessments, the worst kind of failure for a compliance tool.
* **Unified text before consolidated text:** Kancelaria Sejmu updates the unified text after each amendment. An official consolidated text only appears every year or two, so it can miss recent amendments.
* **Consolidated text kept as a fallback:** it covers an act that has consolidated texts but no unified text, at the cost of one extra metadata request.

## Consequences

**Positive:**
* Assessments use the current wording of amended acts, as the PoC scope requires.

**Negative / accepted trade-offs:**
* For amended acts the main path is PDF extraction rather than HTML parsing. PDFs bring page headers and footers (`©Kancelaria Sejmu`, `s. 1/48`, generation dates) and words hyphenated across lines. Task 2.3 has to clean these up reliably.
* The unified text is an informational copy, not the legally binding publication. For a PoC that assesses compliance with current law this is acceptable, and `source_url` points at the exact file used.
* The unified text already includes amendments that have been published but are not yet in force, so text extraction (Task 2.3) must filter them out. Verified on 2026-11-21 against the `U` PDF of `DU/2014/827` (dated 2026-10-29) and the amending act `DU/2026/1353` (in force from 2027-09-27):
  * future wording is wrapped in `<…>`, e.g. `<5ga) aktualizacja oprogramowania – …>`, with a footnote such as "Dodany pkt 5ga w art. 2 wejdzie w życie z dn. 27.09.2027 r. (Dz. U. z 2026 r. poz. 1353)";
  * wording in force today but due to be replaced or repealed is wrapped in `[…]`, directly followed by its `<…>` replacement where there is one;
  * the official consolidated text as published (`O` PDF of `DU/2026/1244`) does not contain these amendments. Its `U` variant, also maintained by Kancelaria Sejmu, does.

  For the state "na dzień dzisiejszy", extraction keeps the content of `[…]` (dropping the brackets) and removes `<…>` blocks together with their content. This is simpler than modelling dates of entry into force, which the PoC leaves out (section 1.3).

## Alternatives considered

* **HTML first (original SPEC):** easier to parse, but outdated for any amended act. Rejected.
* **Newest official consolidated text first:** binding, but it can lag behind recent amendments. Kept as the fallback.
* **Applying amending acts to the original text ourselves:** far beyond the PoC scope.
