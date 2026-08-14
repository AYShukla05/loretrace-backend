import asyncio

from app.models.enums import ScrapeJobStatus
from app.models.source import Source
from app.worker import runner


def run(coro):
    return asyncio.run(coro)


class FakeSession:
    def __init__(self):
        self.executed = []
        self.added = []
        self.commits = 0

    async def execute(self, stmt):
        self.executed.append(stmt)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


class FakeSessionCM:
    def __init__(self, session: FakeSession):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc_info):
        return False


class FakeEngine:
    def __init__(self):
        self.disposed = 0

    async def dispose(self):
        self.disposed += 1


def make_source(source_id: int = 1) -> Source:
    return Source(id=source_id, url="https://example.com/book", added_by=1)


def test_returns_false_when_queue_is_empty(monkeypatch):
    monkeypatch.setattr(runner, "async_session", lambda: FakeSessionCM(FakeSession()))

    async def fake_claim(db):
        return None

    monkeypatch.setattr(runner, "claim_next_source", fake_claim)

    assert run(runner._run_one_watched(watchdog_timeout=1.0)) is False


def test_fast_job_completes_without_watchdog_intervention(monkeypatch):
    sessions = iter([FakeSession(), FakeSession()])
    monkeypatch.setattr(runner, "async_session", lambda: FakeSessionCM(next(sessions)))

    source = make_source()

    async def fake_claim(db):
        return source

    ran = []

    async def fake_run_claimed_job(claimed_source, db):
        ran.append(claimed_source)

    fake_engine = FakeEngine()
    monkeypatch.setattr(runner, "claim_next_source", fake_claim)
    monkeypatch.setattr(runner, "run_claimed_job", fake_run_claimed_job)
    monkeypatch.setattr(runner, "engine", fake_engine)

    assert run(runner._run_one_watched(watchdog_timeout=1.0)) is True
    assert ran == [source]
    assert fake_engine.disposed == 0


def test_hung_job_is_abandoned_and_source_marked_failed(monkeypatch):
    # Simulates the real bug this watchdog exists for: a worker iteration
    # that never returns and never raises. asyncio.sleep is used as the
    # stand-in hang (unlike the real bug, it IS cleanly cancellable, but the
    # watchdog is deliberately designed not to depend on that — see
    # app/worker/runner.py's module docstring comment).
    fake_sessions = [FakeSession(), FakeSession(), FakeSession()]
    sessions = iter(fake_sessions)
    monkeypatch.setattr(runner, "async_session", lambda: FakeSessionCM(next(sessions)))

    source = make_source(source_id=42)

    async def fake_claim(db):
        return source

    async def fake_run_claimed_job(claimed_source, db):
        await asyncio.sleep(3600)

    fake_engine = FakeEngine()
    monkeypatch.setattr(runner, "claim_next_source", fake_claim)
    monkeypatch.setattr(runner, "run_claimed_job", fake_run_claimed_job)
    monkeypatch.setattr(runner, "engine", fake_engine)

    result = run(runner._run_one_watched(watchdog_timeout=0.05))

    assert result is True
    assert fake_engine.disposed == 1

    recovery_session = fake_sessions[2]
    assert recovery_session.commits == 1
    job = recovery_session.added[0]
    assert job.source_id == 42
    assert job.status == ScrapeJobStatus.FAILED
    assert job.error_message is not None
    assert len(recovery_session.executed) == 1
