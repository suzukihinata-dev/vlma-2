import pytest

from vlma.actions import Action, ParseFailure, parse_action

CANONICAL = [
    "CONSTRUCT e : perp a b c e [002] cong b c b e [003]",
    "CONSTRUCT e :",
    "APPLY r111 : eqangle b c c e c e b e [005] from [003]",
    "APPLY AR : cong b d b e [007] from [003] [001]",
    "APPLY r28 : coll a d e [014] from [013]",
    "APPLY AR : rconst e g f g 1/2 [058] from [021] [057]",  # 比の定数
    "APPLY AR : aconst a b c d 2pi/3 [060] from [021]",  # 角の定数
]


@pytest.mark.parametrize("text", CANONICAL)
def test_format_parse_roundtrip(text):
    action = parse_action(text)
    assert isinstance(action, Action)
    assert action.format() == text


def test_constant_is_not_a_point():
    a = parse_action(CANONICAL[5])
    assert a.args == ("e", "g", "f", "g", "1/2") and a.point_args == ("e", "g", "f", "g")


def test_whitespace_is_tolerated():
    a = parse_action("  APPLY r111 :\n eqangle b c c e c e b e [005]   from [003] ")
    assert isinstance(a, Action) and a.format() == CANONICAL[2]


def test_genesis_strings():
    assert parse_action(CANONICAL[0]).to_genesis() == "x00 e : perp a b c e [002] cong b c b e [003]"
    assert parse_action(CANONICAL[2]).to_genesis() == "eqangle b c c e c e b e [005] r111 [003]"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        "CONSTRUCT",
        "CONSTRUCT E : perp a b c e [002]",  # 点名は小文字
        "CONSTRUCT e : perp a b c e",  # 番号が無い
        "APPLY r111 eqangle b c c e [005]",  # コロンが無い
        "APPLY r111 : eqangle b c c e",  # 結論の番号が無い
        "APPLY r111 : eqangle b c c e [005] from",  # from の後に根拠が無い
        "APPLY r111 : eqangle b c c e [005] extra",  # 余計な語
        "APPLY r111 : eqangle b c c e [005]\nAPPLY r111 : cong a b a b [006]",  # 2手を1回で出した
    ],
)
def test_malformed_is_parse_failure(text):
    assert isinstance(parse_action(text), ParseFailure)


def test_parse_facts_accepts_constants_that_genesisgeo_skips():
    from vlma.actions import parse_facts

    text = "a : ; b : ; c : rconst a b a c 1/2 [000] ; d : para a c b d [001] eqangle a b a d a d a c [002] ;"
    assert parse_facts(text) == [
        ("rconst", ("a", "b", "a", "c", "1/2"), "000"),
        ("para", ("a", "c", "b", "d"), "001"),
        ("eqangle", ("a", "b", "a", "d", "a", "d", "a", "c"), "002"),
    ]
    assert parse_facts("e : aconst a b c d 2pi/3 [007]") == [("aconst", ("a", "b", "c", "d", "2pi/3"), "007")]
