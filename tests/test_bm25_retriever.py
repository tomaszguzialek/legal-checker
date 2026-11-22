import pytest

from legal_checker.bm25_retriever import stem, tokenize


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
