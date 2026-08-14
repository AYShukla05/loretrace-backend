import asyncio

from app.models.enums import ScrapeJobStatus, SourceStatus
from app.models.source import Source
from app.worker import queue


def run(coro):
    return asyncio.run(coro)


class FakeSession:
    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


def make_source(source_id: int = 1) -> Source:
    return Source(id=source_id, url="https://example.com/book", added_by=1)


def test_run_claimed_job_marks_completed_on_success(monkeypatch):
    source = make_source()
    db = FakeSession()

    async def fake_process_source(src, session):
        return True

    monkeypatch.setattr(queue, "process_source", fake_process_source)

    run(queue.run_claimed_job(source, db))

    assert source.status == SourceStatus.COMPLETED
    assert source.last_scraped_at is not None
    assert db.commits == 2  # job-insert commit, then the finalize commit
    assert source in db.added  # re-attached: it was loaded via a different session
    assert db.added[-1].status == ScrapeJobStatus.COMPLETED


def test_run_claimed_job_skips_last_scraped_at_when_content_unchanged(monkeypatch):
    source = make_source()
    db = FakeSession()

    async def fake_process_source(src, session):
        return False

    monkeypatch.setattr(queue, "process_source", fake_process_source)

    run(queue.run_claimed_job(source, db))

    assert source.status == SourceStatus.COMPLETED
    assert source.last_scraped_at is None
    assert source.last_checked_at is not None


def test_run_claimed_job_marks_failed_on_exception(monkeypatch):
    source = make_source()
    db = FakeSession()

    async def fake_process_source(src, session):
        raise RuntimeError("boom")

    monkeypatch.setattr(queue, "process_source", fake_process_source)

    run(queue.run_claimed_job(source, db))

    assert source.status == SourceStatus.FAILED
    assert source in db.added  # re-attached: it was loaded via a different session
    assert db.added[-1].status == ScrapeJobStatus.FAILED
    assert db.added[-1].error_message == "boom"
    # The failure path still has to persist that failure, not leave it hung.
    assert db.commits == 2
