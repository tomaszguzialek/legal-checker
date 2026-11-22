"""In-memory BM25 retrieval over an act's units (SPEC Task 3.2)."""

import re

# Words and numbers; punctuation separates tokens ("art. 27a," -> "art", "27a").
_WORD = re.compile(r"\w+")
# Frequent function words that carry no meaning for retrieval.
STOPWORDS = frozenset(
    "a albo ani aby by być czy dla do i ich jak jako jest jego jej już lub na nad o od oraz po pod"
    " przez przy się są ta tak te tego to tym u w we z za ze że który która które których którym".split()
)
# Inflectional and common derivational endings, longest first so that "ami" wins over "i".
# This is a light suffix stripper, not a full Polish stemmer: it maps the forms that matter
# for retrieval ("odstąpienie", "odstąpienia", "odstąpić", "odstąpił") to one stem ("odstąp").
_SUFFIXES = sorted(
    """
    ieniem aniem eniem owania owanie owaniu
    ienia ienie ieniu ania anie aniu enia enie eniu ości ość ować owej owym owych owie owi ował
    ach ami ego emu ich imi iej ymi ych ową owa owe owy ały ało ała ali ili ują uje
    om ów ie ej ym im ią ię em ał ił ić ać eć ą ę
    a e i o u y
    """.split(),
    key=len,
    reverse=True,
)
# Stems shorter than this are not trimmed further: "dni" stays rather than becoming "dn".
_MIN_STEM = 3


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens of Polish text, without stopwords, each reduced to its stem."""
    return [stem(word) for word in _WORD.findall(text.lower()) if word not in STOPWORDS]


def stem(word: str) -> str:
    """Strip the longest known ending that leaves at least ``_MIN_STEM`` characters; numbers stay.

    "ó" becomes "o", as it alternates within a word's forms ("umowa" / "umów", "sposób" / "sposobu").
    """
    if not word.isalpha():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            word = word[: -len(suffix)]
            break
    return word.replace("ó", "o")
