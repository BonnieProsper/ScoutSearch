# scout/search/query.py

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedQuery:
    required: set[str]
    optional: set[str]
    exclude: set[str]
    phrases: list[list[str]]
    has_or: bool


def parse_query(query: str) -> ParsedQuery:
    tokens = query.strip().split()

    required: set[str] = set()
    optional: set[str] = set()
    exclude: set[str] = set()
    phrases: list[list[str]] = []

    current_phrase: list[str] | None = None
    saw_or = False
    has_or = False

    for raw in tokens:
        token = raw.lower()

        if token == "or":
            saw_or = True
            has_or = True
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
        has_or=has_or,
    )
