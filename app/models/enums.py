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


# SQLAlchemy's Enum stores the member *name* by default (e.g. "PENDING"), not
# its value ("pending"). values_callable makes it store the value instead, to
# match the migration and the spec's lowercase status strings.
def pg_enum(enum_cls: type[enum.Enum]) -> SAEnum:
    return SAEnum(enum_cls, native_enum=False, values_callable=lambda obj: [e.value for e in obj])
