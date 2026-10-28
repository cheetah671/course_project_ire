"""HotpotQA and DPR spell some Wikipedia titles differently (Ada job 44382): the title key must unify them."""
import unicodedata

import pandas as pd

from ragroute.data.build_corpus import norm_title, title_key


def test_unicode_forms_match():
    composed = unicodedata.normalize("NFC", "Václav Havel")
    decomposed = unicodedata.normalize("NFD", "Václav Havel")
    assert composed != decomposed
    assert norm_title(decomposed) == norm_title(composed) == composed


def test_html_entities_and_underscores():
    assert norm_title("Army &amp; Navy Stores (United Kingdom)") == "Army & Navy Stores (United Kingdom)"
    assert norm_title('Zachary &quot;Skeeter&quot; Reece') == 'Zachary "Skeeter" Reece'
    assert norm_title(" Three_card_brag ") == "Three card brag"


def test_series_key_matches_scalar_key():
    titles = ["Anaïs Nin", unicodedata.normalize("NFD", "La bohème"), "Richard Hornsby &amp; Sons"]
    assert title_key(pd.Series(titles)).tolist() == [norm_title(t) for t in titles]
