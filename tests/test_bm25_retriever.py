import pytest

from legal_checker.bm25_retriever import InMemoryBm25Index, stem, tokenize
from legal_checker.models import LegalUnit


def test_tokenize_lowercases_and_drops_punctuation_and_stopwords():
    tokens = tokenize("Konsument, który zawarł umowę na odległość, może w terminie 14 dni od niej odstąpić (art. 27a).")

    assert tokens == ["konsument", "zawarł", "umow", "odległ", "moż", "termin", "14", "dni", "niej", "odstąp", "art", "27a"]


@pytest.mark.parametrize(
    "forms",
    [
        ["odstąpienie", "odstąpienia", "odstąpieniu", "odstąpić", "odstąpił"],
        ["konsument", "konsumenta", "konsumentowi", "konsumentem", "konsumentów"],
        ["umowa", "umowy", "umowie", "umową", "umowę", "umów"],
        ["przedsiębiorca", "przedsiębiorcy", "przedsiębiorcę", "przedsiębiorców"],
        ["odległość", "odległości"],
        ["sposób", "sposobu"],
    ],
)
def test_stem_maps_inflected_forms_to_one_stem(forms):
    assert len({stem(form) for form in forms}) == 1


def test_stem_keeps_short_words_and_numbers():
    assert [stem("dni"), stem("art"), stem("27a"), stem("449")] == ["dni", "art", "27a", "449"]


def unit(article: str, text: str, hierarchy_path: list[str] | None = None) -> LegalUnit:
    return LegalUnit(
        unit_id=f"DU/2014/827:art.{article}",
        eli="DU/2014/827",
        article=article,
        hierarchy_path=[*(hierarchy_path or []), f"Art. {article}"],
        text=text,
    )


def test_index_tokenizes_each_unit_with_its_parent_headings():
    units = [
        unit("1", "Ustawa określa prawa konsumenta."),
        unit("27", "Art. 27. Konsument może odstąpić od umowy.", ["Rozdział 4 Prawo odstąpienia"]),
    ]

    index = InMemoryBm25Index(units)

    assert index.units == units
    assert index.corpus[1] == [*tokenize("Rozdział 4 Prawo odstąpienia Art. 27"), "art", "27", "konsument", "moż", "odstąp", "umow"]
    assert index.bm25.corpus_size == 2


def test_index_rejects_an_act_without_units():
    with pytest.raises(ValueError):
        InMemoryBm25Index([])
