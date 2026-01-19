# scout/search/query.py

from dataclasses import dataclass
from typing import List, Set


@dataclass(frozen=True)
class ParsedQuery:
    include: Set[str]
    exclude: Set[str]
    optional: Set[str]
    phrases: List[List[str]]


def parse_query(query: str) -> ParsedQuery:
    tokens = query.lower().split()

    include = set()
    exclude = set()
    optional = set()
    phrases = []

    current_phrase = None

    for token in tokens:
        if token.startswith('"'):
            current_phrase = [token.lstrip('"')]
        elif token.endswith('"') and current_phrase is not None:
            current_phrase.append(token.rstrip('"'))
            phrases.append(current_phrase)
            current_phrase = None
        elif current_phrase is not None:
            current_phrase.append(token)
        elif token.startswith("-"):
            exclude.add(token[1:])
        elif token.upper() == "OR":
            continue
        else:
            include.add(token)

    return ParsedQuery(
        include=include,
        exclude=exclude,
        optional=optional,
        phrases=phrases,
    )
