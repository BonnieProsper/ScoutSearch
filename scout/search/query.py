# scout/search/query.py

from dataclasses import dataclass
from typing import List, Set


@dataclass(frozen=True)
class ParsedQuery:
    required: Set[str]
    optional: Set[str]
    exclude: Set[str]
    phrases: List[List[str]]


def parse_query(query: str) -> ParsedQuery:
    tokens = query.strip().split()

    required: Set[str] = set()
    optional: Set[str] = set()
    exclude: Set[str] = set()
    phrases: List[List[str]] = []

    current_phrase: List[str] | None = None
    saw_or = False

    for raw in tokens:
        token = raw.lower()

        if token == "or":
            saw_or = True
            continue

        if token.startswith('"'):
            current_phrase = [token.lstrip('"')]
            continue

        if token.endswith('"') and current_phrase is not None:
            current_phrase.append(token.rstrip('"'))
            phrases.append(current_phrase)
            current_phrase = None
            continue

        if current_phrase is not None:
            current_phrase.append(token)
            continue

        if token.startswith("-"):
            exclude.add(token[1:])
            continue

        if saw_or:
            optional.add(token)
            saw_or = False
        else:
            required.add(token)

    return ParsedQuery(
        required=required,
        optional=optional,
        exclude=exclude,
        phrases=phrases,
    )
