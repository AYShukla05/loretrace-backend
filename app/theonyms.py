"""Query-side theonym expansion.

Public-domain translations within one tradition do not agree on what to
call a god. Butler's Iliad says "Venus" and "Jove" where Evelyn-White's
Hesiod says "Aphrodite" and "Zeus"; Ovid (a Roman source in a Greek
corpus) uses the Roman names throughout. A user asking about "Aphrodite"
in Greek-name vocabulary embeds far from the Iliad passage that actually
answers them, so the retrieval never surfaces the disagreement between
the two texts.

Expanding the query with the equivalent names before embedding closes
that gap. See LoreTrace_Greek_Retrieval_Characterization.md: appending
the Roman equivalents moves the Iliad's "Venus is child to Jove" chunk
from rank 86 to rank 1 for the naive "parents of Aphrodite" query.

Deterministic, no LLM. Scoped per tradition so a Roman name in a query
that was never about Greek myth is left alone.
"""

import re

# Each tuple groups the names that refer to one figure across a
# tradition's own texts and its later reception in other languages. The
# first entry is the tradition's own primary name; it anchors the
# appended clause. Keyed by lowercased tradition, matching how
# app.schemas.source.normalize_tradition stores it (title-cased) once
# lowercased for lookup.
_THEONYM_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "greek": (
        ("Aphrodite", "Venus", "Cytherea", "Cypris"),
        ("Zeus", "Jupiter", "Jove"),
        ("Hera", "Juno"),
        ("Poseidon", "Neptune"),
        ("Ares", "Mars"),
        ("Athena", "Athene", "Minerva", "Pallas"),
        ("Artemis", "Diana"),
        ("Hermes", "Mercury"),
        ("Hephaestus", "Hephaistos", "Vulcan"),
        ("Dionysus", "Dionysos", "Bacchus"),
        ("Demeter", "Ceres"),
        ("Persephone", "Proserpina", "Proserpine", "Kore"),
        ("Hades", "Pluto", "Dis"),
        ("Cronus", "Cronos", "Kronos", "Saturn"),
        ("Uranus", "Ouranos"),
        ("Gaia", "Gaea", "Terra"),
        ("Rhea", "Ops"),
        ("Leto", "Latona"),
        ("Hestia", "Vesta"),
        ("Eros", "Cupid"),
        ("Heracles", "Herakles", "Hercules"),
        ("Odysseus", "Ulysses"),
        ("Helios", "Sol"),
        ("Selene", "Luna"),
        ("Eos", "Aurora"),
    ),
}


def _mentions(name: str, query: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", query, re.IGNORECASE) is not None


def expand_query(query: str, tradition: str | None) -> str:
    """Append equivalent-name clauses for any grouped theonym the query
    mentions. Returns the query unchanged when the tradition has no table
    or no grouped name appears.
    """
    if tradition is None:
        return query
    groups = _THEONYM_GROUPS.get(tradition.lower())
    if groups is None:
        return query

    clauses: list[str] = []
    for group in groups:
        if any(_mentions(name, query) for name in group):
            primary, *rest = group
            clauses.append(f"{primary} (also called {', '.join(rest)})")

    if not clauses:
        return query
    return f"{query} {'; '.join(clauses)}."
