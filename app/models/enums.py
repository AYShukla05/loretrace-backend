import enum

from sqlalchemy import Enum as SAEnum


class SourceType(str, enum.Enum):
    GUTENBERG_TEXT = "gutenberg_text"
    WIKISOURCE = "wikisource"
    WIKIPEDIA = "wikipedia"
    MANUAL_UPLOAD = "manual_upload"


class SourceStatus(str, enum.Enum):
    PENDING = "pending"
    SCRAPING = "scraping"
    COMPLETED = "completed"
    FAILED = "failed"


class ScrapeJobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Era(str, enum.Enum):
    PRE_COLONIAL = "pre_colonial"
    COLONIAL_ERA = "colonial_era"
    POST_INDEPENDENCE = "post_independence"
    CONTEMPORARY = "contemporary"


class AuthorPosition(str, enum.Enum):
    INDIGENOUS_PRIMARY_TEXT = "indigenous_primary_text"
    INDIGENOUS_SCHOLAR = "indigenous_scholar"
    COLONIAL_ADMINISTRATOR = "colonial_administrator"
    MISSIONARY = "missionary"
    WESTERN_ACADEMIC = "western_academic"
    UNKNOWN_COMPILER = "unknown_compiler"


class TextRole(str, enum.Enum):
    PRIMARY_TRANSLATION = "primary_translation"
    SECONDARY_COMMENTARY = "secondary_commentary"
    TERTIARY_SUMMARY = "tertiary_summary"


# Independent of AuthorPosition, see LoreTrace_Credibility_Suggestion_Design.md
# section 4.1 (Max Muller vs. David Frawley: foreign-born doesn't imply
# textual-study-only, origin and lived practice don't always travel together).
class HistoriographicalMethod(str, enum.Enum):
    ORAL_TRADITION = "oral_tradition"
    TEXTUAL_CRITICAL = "textual_critical"
    COLONIAL_COMPARATIVE_MYTHOLOGY = "colonial_comparative_mythology"
    ARCHAEOLOGICAL_CORRELATION = "archaeological_correlation"
    ARCHAEOASTRONOMICAL_DATING = "archaeoastronomical_dating"
    GENETIC_ANTHROPOLOGICAL = "genetic_anthropological"
    MODERN_ACADEMIC_CONSENSUS = "modern_academic_consensus"
    UNSPECIFIED = "unspecified"


class AuthorOrigin(str, enum.Enum):
    INDIGENOUS_BORN = "indigenous_born"
    FOREIGN_BORN = "foreign_born"
    UNKNOWN = "unknown"


class AuthorEpistemicBasis(str, enum.Enum):
    LIVED_PRACTICE = "lived_practice"
    TEXTUAL_STUDY_ONLY = "textual_study_only"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class CredibilityEntityType(str, enum.Enum):
    AUTHOR = "author"
    INSTITUTION = "institution"


class SuggestionStatus(str, enum.Enum):
    PENDING = "pending"
    REVIEWED = "reviewed"


# SQLAlchemy's Enum stores the member *name* by default (e.g. "PENDING"), not
# its value ("pending"). values_callable makes it store the value instead, to
# match the migration and the spec's lowercase status strings.
def pg_enum(enum_cls: type[enum.Enum]) -> SAEnum:
    return SAEnum(enum_cls, native_enum=False, values_callable=lambda obj: [e.value for e in obj])
