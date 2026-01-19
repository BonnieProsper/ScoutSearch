from scout.search.query import parse_query


def test_parse_required_and_exclude():
    q = parse_query("quick fox -lazy")
    assert q.required == {"quick", "fox"}
    assert q.exclude == {"lazy"}


def test_parse_or_optional():
    q = parse_query("quick OR fox")
    assert q.required == {"quick"}
    assert q.optional == {"fox"}


def test_parse_phrase():
    q = parse_query('"quick brown fox"')
    assert q.phrases == [["quick", "brown", "fox"]]
