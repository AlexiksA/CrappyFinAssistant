"""
Full CrappyFinAssist pipeline - refreshes every phase, in order, and
delivers the resulting alerts. Built to run unattended from cron.

Stages (each isolated: a failed stage is logged and the rest still run
on whatever data they have):
  1. portfolio  - newest Revolut statement from Drive -> company table
  2. prices     - yfinance history -> technical signals (price_data.py)
  3. news       - discover new articles (Google News RSS, DDGS fallback)
  4. sentiment  - classify new articles; hybrid FinBERT+LLM when Ollama is
                  reachable, plain FinBERT otherwise (results are cached)
  5. alerts     - alert engine + Telegram digest, including any alerts a
                  previous run failed to deliver

Safety:
  - only one run at a time (lock file; a second run exits quietly)
  - chdirs to its own folder, so cron can call it from anywhere
  - never starts the interactive Drive login - an unusable token just
    skips the portfolio stage
  - forces a headless browser when there's no display
  - everything goes to logs/pipeline.log (rotated, 5 x 5 MB)
  - a stage failure sends one Telegram warning, at most every 6 hours

Usage:
    python pipeline.py              # full run, sends the digest
    python pipeline.py --no-send    # full run, prints the digest instead;
                                    # alerts stay queued for the next real run
"""

import fcntl
import hashlib
import io
import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
LOCK_FILE = os.path.join(LOG_DIR, "pipeline.lock")
FAILURE_NOTICE_FILE = os.path.join(LOG_DIR, ".last_failure_notice")
PORTFOLIO_HASH_FILE = os.path.join(LOG_DIR, ".last_statement_hash")
FAILURE_NOTICE_INTERVAL = 6 * 3600

logger = logging.getLogger("pipeline")


class StageSkipped(Exception):
    """A stage deliberately didn't run - not a failure."""


class _LogStream(io.TextIOBase):
    """Stands in for stdout/stderr: every line the project code prints ends
    up in the rotating log, and is echoed to the terminal when there is one."""

    def __init__(self, level: int, echo=None):
        self.level = level
        self.echo = echo
        self._buffer = ""

    def write(self, text: str) -> int:
        if self.echo is not None:
            self.echo.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                logger.log(self.level, line)
        return len(text)

    def flush(self) -> None:
        if self.echo is not None:
            self.echo.flush()


def _setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # a logging error must never write to stderr, which now loops back here
    logging.raiseExceptions = False

    interactive = sys.stdout.isatty()
    sys.stdout = _LogStream(logging.INFO, sys.__stdout__ if interactive else None)
    sys.stderr = _LogStream(logging.WARNING, sys.__stderr__ if interactive else None)


def _acquire_lock():
    handle = open(LOCK_FILE, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def ollama_available() -> bool:
    """Reachable *and* has the configured model pulled."""
    import requests

    from sentiment_digest import DEF_ENDPOINT, DEF_MODEL

    base = DEF_ENDPOINT.split("/api/")[0]
    try:
        response = requests.get(f"{base}/api/tags", timeout=5)
        response.raise_for_status()
        names = {m.get("name") for m in response.json().get("models", [])}
    except Exception as e:
        print(f"Ollama not reachable ({e}) - using FinBERT only")
        return False
    if DEF_MODEL not in names:
        print(f"Ollama is up but {DEF_MODEL} isn't pulled - using FinBERT only")
        return False
    return True


def _file_hash(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _read_state(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return handle.read().strip() or None


def _write_state(path: str, value: str) -> None:
    with open(path, "w") as handle:
        handle.write(value)


# --- stages ---------------------------------------------------------------


def stage_portfolio() -> None:
    import pdf_from_drive
    from google.oauth2.credentials import Credentials

    # pdf_from_drive falls back to an interactive browser login when the
    # token is unusable; under cron that would hang until the timeout
    if not os.path.exists("token.json"):
        raise StageSkipped(
            "token.json missing - run `python pdf_from_drive.py` once by hand"
        )
    creds = Credentials.from_authorized_user_file("token.json", pdf_from_drive.SCOPES)
    if not creds.valid and not (creds.expired and creds.refresh_token):
        raise StageSkipped(
            "Drive token can't be refreshed - run `python pdf_from_drive.py` by hand"
        )

    from google.auth.exceptions import RefreshError

    try:
        statement = pdf_from_drive.get_latest_stat()
    except RefreshError as e:
        raise RuntimeError(
            "Drive login expired - run `python pdf_from_drive.py` by hand to log in "
            "again. If this comes back every ~7 days, the Google Cloud OAuth app is "
            "in Testing mode: publish it to Production to stop refresh tokens expiring."
        ) from e
    if statement is None:
        raise RuntimeError("could not download the latest statement from Drive")

    # only re-parse when the statement actually changed; compared against
    # the last *successfully processed* one, so a failed parse is retried
    statement_hash = _file_hash("stats.pdf")
    if statement_hash == _read_state(PORTFOLIO_HASH_FILE):
        print("statement unchanged since the last successful import")
        return

    import pdf_extract

    # upserts current holdings and drops the ones no longer in the statement
    result = pdf_extract.import_statement("stats.pdf")

    _write_state(PORTFOLIO_HASH_FILE, statement_hash)
    removed = ", ".join(result["removed"]) or "none"
    print(
        f"portfolio updated from new statement ({', '.join(result['imported'])}); "
        f"removed: {removed}"
    )


def stage_prices() -> None:
    import sqlite3

    import price_data

    started = datetime.now().isoformat()
    price_data.price_data_main()

    # price_data quietly returns early if yfinance gives it nothing; don't
    # let the alert stage run on stale signals without saying so
    conn = sqlite3.connect("data/hist.db")
    try:
        latest = conn.execute("SELECT MAX(last_update) FROM signal").fetchone()[0]
    finally:
        conn.close()
    if latest is None or latest < started:
        raise RuntimeError("no fresh signals were written - yfinance may be down")


def stage_news() -> None:
    from sentiment_digest import discover_news_for_tickers

    try:
        found = discover_news_for_tickers(backend="google")
        backend = "google"
    except Exception as e:
        print(f"Google News discovery failed ({e}) - falling back to DDGS")
        found = discover_news_for_tickers(backend="ddgs")
        backend = "ddgs"

    total = sum(len(articles) for articles in found.values())
    print(f"news discovery ({backend}): {total} new article(s)")
    for ticker, articles in found.items():
        print(f"  {ticker}: {len(articles)}")


def stage_sentiment(use_hybrid: bool) -> list[dict]:
    from sentiment_digest import _warm_up_model, summarize_sentiment

    if use_hybrid:
        _warm_up_model()
        try:
            return summarize_sentiment(use_hybrid=True)
        except Exception as e:
            print(f"hybrid classification failed ({e}) - retrying with FinBERT only")
    return summarize_sentiment(use_hybrid=False)


def stage_alerts(summary: list[dict] | None, with_ai: bool, send: bool) -> None:
    import send_mes
    from alert_engine import pending_alerts, run_alert_engine

    queued = pending_alerts()
    if queued:
        print(f"{len(queued)} undelivered alert(s) from earlier runs")

    # an empty list (not None) when sentiment failed: None would make the
    # engine re-run classification itself; [] just means technicals only
    alerts = run_alert_engine(sentiment_summary=summary if summary is not None else [])
    print(f"{len(alerts)} new alert(s)")
    batch = queued + alerts

    if not send:
        if batch:
            print(send_mes.format_digest(batch, summary, with_ai))
        print("--no-send: nothing delivered, alerts stay queued for the next run")
        return

    if not send_mes.send_alerts(batch, sentiment_summary=summary, with_ai=with_ai):
        raise RuntimeError("Telegram delivery failed - alerts stay queued")


# --- orchestration --------------------------------------------------------


def _run_stage(name: str, results: dict, fn, *args):
    print(f"===== {name} =====")
    started = time.time()
    try:
        value = fn(*args)
        results[name] = ("ok", time.time() - started, "")
        return value
    except StageSkipped as e:
        print(f"{name} skipped: {e}")
        results[name] = ("skipped", time.time() - started, str(e))
    except Exception as e:
        logger.exception(f"{name} failed")
        results[name] = ("failed", time.time() - started, f"{type(e).__name__}: {e}")
    return None


def _notify_failures(results: dict) -> None:
    failed = {name: info for name, info in results.items() if info[0] == "failed"}
    if not failed:
        return

    last = _read_state(FAILURE_NOTICE_FILE)
    if last and time.time() - float(last) < FAILURE_NOTICE_INTERVAL:
        print("stage failure notice already sent recently - not repeating it")
        return

    import html

    import send_mes

    lines = ["⚠️ <b>CrappyFinAssist</b> pipeline problem"]
    for name, (_, _, error) in failed.items():
        lines.append(f"• <b>{name}</b>: {html.escape(error[:300])}")
    lines.append(
        f"Details: <code>{html.escape(os.path.join(PROJECT_DIR, LOG_FILE))}</code>"
    )

    try:
        if send_mes.send_message("\n".join(lines)):
            _write_state(FAILURE_NOTICE_FILE, str(time.time()))
    except Exception as e:
        print(f"could not send the failure notice ({e})")


def main() -> int:
    _setup_logging()
    send = "--no-send" not in sys.argv

    lock = _acquire_lock()
    if lock is None:
        print("another pipeline run is still in progress - exiting")
        return 0

    started = time.time()
    try:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            os.environ["HARNESS_HEADLESS"] = "1"

        use_hybrid = ollama_available()
        print(
            f"pipeline start | mode={'hybrid' if use_hybrid else 'finbert'} "
            f"| headless={os.environ.get('HARNESS_HEADLESS') == '1'} | send={send}"
        )

        results: dict = {}
        _run_stage("portfolio", results, stage_portfolio)
        _run_stage("prices", results, stage_prices)
        _run_stage("news", results, stage_news)
        summary = _run_stage("sentiment", results, stage_sentiment, use_hybrid)
        _run_stage("alerts", results, stage_alerts, summary, use_hybrid, send)

        print("===== summary =====")
        for name, (status, seconds, detail) in results.items():
            print(f"{name:<10} {status:<8} {seconds:7.1f}s  {detail}")
        print(f"total {time.time() - started:.1f}s")

        if send:
            _notify_failures(results)

        return 1 if any(info[0] == "failed" for info in results.values()) else 0
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
