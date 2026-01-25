from scout.index.tokens import Tokenizer


def test_unicode_tokenization():
    t = Tokenizer()
    tokens = t.tokenize("naïve café")
    assert "naïve" in tokens
    assert "café" in tokens


def test_punctuation_tokenization():
    t = Tokenizer()
    tokens = t.tokenize("C++ developer")
    assert "c" in tokens
    assert "developer" in tokens
