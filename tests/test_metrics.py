from ragroute.eval.metrics import (answer_contains, exact_match, extract_answer, normalize_answer, parse_verdict,
                                  score_row, token_f1)


def test_normalize_squad_style():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("an Apple, a day") == "apple day"


def test_em_f1_multi_gold():
    golds = ["Barack Obama", "Obama"]
    assert exact_match("obama.", golds) == 1.0
    assert exact_match("President Obama", golds) == 0.0
    assert abs(token_f1("President Obama", golds) - 2 / 3) < 1e-9  # best gold: "obama": p=1/2, r=1
    assert token_f1("", golds) == 0.0


def test_extract_answer_first_line_and_prefix():
    assert extract_answer("Answer: Paris\nBecause the passage says so") == "Paris"
    assert extract_answer("\n\n  Short answer - 1912 ") == "1912"
    assert extract_answer(None) == ""


def test_contains_is_lenient_only():
    assert answer_contains("It was Paris, France", ["paris"]) == 1.0
    assert exact_match("It was Paris, France", ["paris"]) == 0.0


def test_verdict_parsing():
    assert parse_verdict("SUPPORTS") == "SUPPORTS"
    assert parse_verdict("Refutes.") == "REFUTES"
    assert parse_verdict("Not enough info") == "INVALID"
    assert parse_verdict("SUPPORTS or REFUTES") == "INVALID"
    assert parse_verdict("True, the claim holds") == "SUPPORTS"


def test_score_row_quality_in_unit_interval():
    r = score_row("qa", "Answer: Obama", ["Barack Obama"])
    assert 0 <= r["quality"] <= 1 and r["em"] == 0.0 and r["quality"] == r["f1"]
    v = score_row("verification", "REFUTES", ["REFUTES"])
    assert v["quality"] == 1.0 and v["acc"] == 1.0
