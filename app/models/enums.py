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


# SQLAlchemy's Enum stores the member *name* by default (e.g. "PENDING"), not
# its value ("pending"). values_callable makes it store the value instead, to
# match the migration and the spec's lowercase status strings.
def pg_enum(enum_cls: type[enum.Enum]) -> SAEnum:
    return SAEnum(enum_cls, native_enum=False, values_callable=lambda obj: [e.value for e in obj])
