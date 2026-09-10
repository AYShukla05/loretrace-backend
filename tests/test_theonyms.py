from app.theonyms import expand_query


def test_greek_name_query_gets_roman_equivalents_appended():
    out = expand_query("Who are the parents of Aphrodite?", "Greek")
    assert out.startswith("Who are the parents of Aphrodite?")
    assert "Aphrodite (also called Venus, Cytherea, Cypris)" in out


def test_roman_name_query_gets_greek_equivalent_appended():
    out = expand_query("Who is Venus the daughter of?", "Greek")
    # Clause is anchored on the tradition's own primary name regardless of
    # which member the query used.
    assert "Aphrodite (also called Venus, Cytherea, Cypris)" in out


def test_multiple_groups_each_add_a_clause():
    out = expand_query("Was Aphrodite the daughter of Zeus and Dione?", "Greek")
    assert "Aphrodite (also called Venus, Cytherea, Cypris)" in out
    assert "Zeus (also called Jupiter, Jove)" in out
    assert out.count(";") == 1  # two clauses joined by one semicolon


def test_match_is_case_insensitive():
    assert "Zeus (also called Jupiter, Jove)" in expand_query("tell me about zeus", "Greek")


def test_substring_does_not_match_whole_word_boundary():
    # "Mars" must not fire on "Marsyas"; "Diana" must not fire on "Dianae".
    assert expand_query("the flaying of Marsyas", "Greek") == "the flaying of Marsyas"


def test_query_without_any_theonym_is_unchanged():
    assert expand_query("how was the world created", "Greek") == "how was the world created"


def test_unknown_tradition_is_unchanged():
    assert expand_query("tell me about Thor and Odin", "Norse") == "tell me about Thor and Odin"


def test_none_tradition_is_unchanged():
    assert expand_query("tell me about Aphrodite", None) == "tell me about Aphrodite"


def test_tradition_lookup_is_case_insensitive():
    assert "Jupiter, Jove" in expand_query("about Zeus", "greek")


def test_modern_japanese_name_gets_archaic_forms_appended():
    out = expand_query("How was Amaterasu born?", "Japanese")
    assert out.startswith("How was Amaterasu born?")
    assert "Ama-terasu" in out
    assert "Heaven-Shining-Great-August-Deity" in out


def test_archaic_japanese_form_in_query_anchors_on_modern_name():
    out = expand_query("what did Sosa no wo do to anger the sun goddess", "Japanese")
    assert "Susanoo (also called" in out


def test_japanese_and_greek_tables_do_not_cross_fire():
    # "Amaterasu" is only in the Japanese table; a Greek-scoped query
    # mentioning it gets nothing appended.
    assert expand_query("was Amaterasu like Helios?", "Greek").endswith("Helios (also called Sol).")
    assert "Ama-terasu" not in expand_query("was Amaterasu like Helios?", "Greek")
