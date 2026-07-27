import enum


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
