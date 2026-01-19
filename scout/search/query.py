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
    """
    Parse a query string into logical components.

    Supported:
    - Required terms (default)
    - Optional terms via OR
    - Excluded terms via -
    - Phrase queries via quotes
    """
    tokens = query.lower().split()

    required: Set[str] = set()
    optional: Set[str] = set()
    exclude: Set[str] = set()
    phrases: List[List[str]] = []

    current_phrase: List[str] | None = None
    last_was_or = False

    for raw in tokens:
        token = raw.strip()

        # Phrase start
        if token.startswith('"'):
            current_phrase = [token.lstrip('"')]
            last_was_or = False
            if token.endswith('"') and len(token) > 1:
                current_phrase[-1] = current_phrase[-1].rstrip('"')
                phrases.append(current_phrase)
                current_phrase = None
            continue

        # Phrase continuation / end
        if current_phrase is not None:
            if token.endswith('"'):
                current_phrase.append(token.rstrip('"'))
                phrases.append(current_phrase)
                current_phrase = None
            else:
                current_phrase.append(token)
            continue

        # OR operator
        if token == "or":
            last_was_or = True
            continue

        # Exclusion
        if token.startswith("-"):
            exclude.add(token[1:])
            last_was_or = False
            continue

        # Normal term
        if last_was_or:
            optional.add(token)
            last_was_or = False
        else:
            required.add(token)

    return ParsedQuery(
        required=required,
        optional=optional,
        exclude=exclude,
        phrases=phrases,
    )
