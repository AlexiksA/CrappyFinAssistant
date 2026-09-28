"""
Phase 4: alert/dedup engine.

Combines Phase 2's technical signals (price_data.py -> data/hist.db's
`signal` table) with Phase 3's news sentiment (sentiment_digest.py ->
summarize_sentiment()) into a single stream of alerts.

Trigger rule: a technical signal and overwhelming sentiment each fire
independently - neither requires the other to agree.

Dedup rule: alert only on a *transition* into a triggered state (or a
change of verdict, e.g. bearish -> bullish sentiment). Staying triggered
on a later run is silent; clearing and re-triggering fires again. State
is tracked in this file's own data/alerts.db, separate from Phase 2/3's
own databases.
"""

import json
import sqlite3 as sql
from datetime import datetime, timezone

from price_data import DEF_MAP
from sentiment_digest import DEF_TABLE, elig_tickers, summarize_sentiment

DEF_HIST_DB = "data/hist.db"
DEF_ALERT_DB = "data/alerts.db"

# don't fire a sentiment alert off too few articles - a single confidently
# classified article shouldn't be "overwhelming" on its own
SENTIMENT_ALERT_MIN_ARTICLES = 3

CREATION_QUERY = """
    CREATE TABLE IF NOT EXISTS alert_state (
        ticker TEXT,
        alert_key TEXT,
        last_state TEXT,
        last_checked TEXT,
        PRIMARY KEY (ticker, alert_key)
    )
"""

ALERT_LOG_QUERY = """
    CREATE TABLE IF NOT EXISTS alert_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        alert_key TEXT,
        alert_type TEXT,
        details TEXT,
        payload TEXT,
        fired_at TEXT,
        delivered INTEGER NOT NULL DEFAULT 0
    )
"""

# columns added after the first version of this table shipped; an alerts.db
# created before then needs them backfilled rather than recreated
ALERT_LOG_MIGRATIONS = {
    "payload": "ALTER TABLE alert_log ADD COLUMN payload TEXT",
    "delivered": "ALTER TABLE alert_log ADD COLUMN delivered INTEGER NOT NULL DEFAULT 0",
}


def _ensure_alert_tables(conn: sql.Connection) -> None:
    conn.execute(CREATION_QUERY)
    conn.execute(ALERT_LOG_QUERY)

    existing = {row[1] for row in conn.execute("PRAGMA table_info(alert_log)")}
    for column, statement in ALERT_LOG_MIGRATIONS.items():
        if column not in existing:
            conn.execute(statement)

    conn.commit()


def _yf_ticker(ticker: str) -> str:
    return DEF_MAP.get(ticker, ticker)


def _get_previous_state(conn: sql.Connection, ticker: str, alert_key: str) -> str | None:
    row = conn.execute(
        "SELECT last_state FROM alert_state WHERE ticker = ? AND alert_key = ?",
        (ticker, alert_key),
    ).fetchone()
    return row[0] if row else None


def _set_state(
    conn: sql.Connection, ticker: str, alert_key: str, state: str | None, timestamp: str
) -> None:
    conn.execute(
        """
        INSERT INTO alert_state (ticker, alert_key, last_state, last_checked)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker, alert_key) DO UPDATE SET
            last_state = excluded.last_state,
            last_checked = excluded.last_checked
        """,
        (ticker, alert_key, state, timestamp),
    )
    conn.commit()


# returns the alert_log row id, which the caller stores on the alert dict so
# delivery (Phase 5) can mark exactly this row once it has actually been sent
def _log_alert(conn: sql.Connection, alert: dict, timestamp: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO alert_log
            (ticker, alert_key, alert_type, details, payload, fired_at, delivered)
        VALUES (?, ?, ?, ?, ?, ?, 0)
        """,
        (
            alert["ticker"],
            alert["alert_key"],
            alert["alert_type"],
            str(alert.get("details", "")),
            json.dumps(alert),
            timestamp,
        ),
    )
    conn.commit()
    return cur.lastrowid


# fires an alert only when current_state is set (something is actually
# triggered) and differs from whatever was stored last run - covers both
# "wasn't triggered, now is" and "was bearish, now bullish"-style flips
def _check_transition(
    conn: sql.Connection,
    ticker: str,
    alert_key: str,
    current_state: str | None,
    timestamp: str,
) -> bool:
    previous_state = _get_previous_state(conn, ticker, alert_key)
    _set_state(conn, ticker, alert_key, current_state, timestamp)
    return current_state is not None and current_state != previous_state


def _technical_signal_alerts(
    conn_alert: sql.Connection, ticker: str, hist_db: str, timestamp: str
) -> list[dict]:
    conn_hist = sql.connect(hist_db)
    try:
        rows = conn_hist.execute(
            "SELECT signal_type, last_value, is_triggered FROM signal WHERE ticker = ?",
            (_yf_ticker(ticker),),
        ).fetchall()
    finally:
        conn_hist.close()

    fired = []
    for signal_type, last_value, is_triggered in rows:
        alert_key = f"technical:{signal_type}"
        current_state = "triggered" if is_triggered else None
        if _check_transition(conn_alert, ticker, alert_key, current_state, timestamp):
            alert = {
                "ticker": ticker,
                "alert_key": alert_key,
                "alert_type": "technical",
                "signal_type": signal_type,
                "value": last_value,
                "details": f"{signal_type} triggered (value={last_value:.4f})",
            }
            alert["log_id"] = _log_alert(conn_alert, alert, timestamp)
            fired.append(alert)

    return fired


def _sentiment_verdict(counts: dict) -> str | None:
    positive, neutral, negative = counts["positive"], counts["neutral"], counts["negative"]
    if positive + neutral + negative < SENTIMENT_ALERT_MIN_ARTICLES:
        return None
    if negative > positive + neutral:
        return "bearish"
    if positive > negative + neutral:
        return "bullish"
    return None


def _sentiment_alert(
    conn_alert: sql.Connection, sentiment_row: dict, timestamp: str
) -> dict | None:
    ticker = sentiment_row["ticker"]
    counts = {
        "positive": sentiment_row["positive"],
        "neutral": sentiment_row["neutral"],
        "negative": sentiment_row["negative"],
    }
    verdict = _sentiment_verdict(counts)
    alert_key = "sentiment"

    if not _check_transition(conn_alert, ticker, alert_key, verdict, timestamp):
        return None

    alert = {
        "ticker": ticker,
        "alert_key": alert_key,
        "alert_type": "sentiment",
        "verdict": verdict,
        "counts": counts,
        "links": sentiment_row.get("links", []),
        "details": f"sentiment turned {verdict} ({counts})",
    }
    alert["log_id"] = _log_alert(conn_alert, alert, timestamp)
    return alert


# top-level entry point. sentiment_summary lets a caller pass in an
# already-computed summarize_sentiment() result to avoid re-running
# classification (FinBERT, or the slower LLM escalation path) on every
# alert check - summarize_sentiment has no persisted label store of its
# own yet, so calling it fresh here every time would be wasteful if this
# is checked more often than new sentiment is actually computed.
def run_alert_engine(
    sentiment_summary: list[dict] | None = None,
    use_hybrid: bool = False,
    hist_db: str = DEF_HIST_DB,
    alert_db: str = DEF_ALERT_DB,
) -> list[dict]:
    if sentiment_summary is None:
        sentiment_summary = summarize_sentiment(use_hybrid=use_hybrid)
    sentiment_by_ticker = {row["ticker"]: row for row in sentiment_summary}

    filtered_df = elig_tickers(DEF_TABLE)
    tickers = list(filtered_df["ticker"]) if filtered_df is not None else []

    conn_alert = sql.connect(alert_db)
    _ensure_alert_tables(conn_alert)
    timestamp = datetime.now(timezone.utc).isoformat()

    alerts: list[dict] = []
    try:
        for ticker in tickers:
            alerts.extend(
                _technical_signal_alerts(conn_alert, ticker, hist_db, timestamp)
            )

            sentiment_row = sentiment_by_ticker.get(ticker)
            if sentiment_row is not None:
                sentiment_alert = _sentiment_alert(conn_alert, sentiment_row, timestamp)
                if sentiment_alert is not None:
                    alerts.append(sentiment_alert)
    finally:
        conn_alert.close()

    return alerts


# --- delivery bookkeeping (used by Phase 5's send_mes.py) ---------------
#
# _check_transition writes the new state before anything is delivered, so a
# failed send would otherwise lose the alert for good: the ticker is already
# recorded as triggered and won't fire again until it clears and re-triggers.
# alert_log therefore keeps every fired alert with delivered = 0 until the
# notifier confirms it went out, and re-reads the leftovers on the next run.


def pending_alerts(alert_db: str = DEF_ALERT_DB) -> list[dict]:
    """Alerts that fired but were never confirmed delivered, oldest first."""
    conn = sql.connect(alert_db)
    _ensure_alert_tables(conn)
    try:
        rows = conn.execute(
            """
            SELECT id, payload FROM alert_log
            WHERE delivered = 0 AND payload IS NOT NULL
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()

    alerts = []
    for log_id, payload in rows:
        try:
            alert = json.loads(payload)
        except json.JSONDecodeError:
            print(f"pending_alerts: skipping unreadable payload for row {log_id}")
            continue
        alert["log_id"] = log_id
        alerts.append(alert)

    return alerts


def latest_alert_batch(alert_db: str = DEF_ALERT_DB) -> list[dict]:
    """Every alert from the most recent run (one run shares a fired_at),
    delivered or not - for previewing the digest without re-firing."""
    conn = sql.connect(alert_db)
    _ensure_alert_tables(conn)
    try:
        rows = conn.execute(
            """
            SELECT id, payload FROM alert_log
            WHERE payload IS NOT NULL
              AND fired_at = (SELECT MAX(fired_at) FROM alert_log)
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()

    alerts = []
    for log_id, payload in rows:
        alert = json.loads(payload)
        alert["log_id"] = log_id
        alerts.append(alert)
    return alerts


def mark_delivered(log_ids: list[int], alert_db: str = DEF_ALERT_DB) -> int:
    """Flag the given alert_log rows as delivered. Returns rows updated."""
    log_ids = [i for i in log_ids if i is not None]
    if not log_ids:
        return 0

    conn = sql.connect(alert_db)
    _ensure_alert_tables(conn)
    try:
        placeholders = ",".join("?" * len(log_ids))
        cur = conn.execute(
            f"UPDATE alert_log SET delivered = 1 WHERE id IN ({placeholders})",
            log_ids,
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


if __name__ == "__main__":
    fired = run_alert_engine()
    print(f"{len(fired)} new alert(s)")
    for alert in fired:
        print(f"[{alert['alert_type']}] {alert['ticker']} -> {alert['details']}")
