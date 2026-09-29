"""
Phase 5: notification delivery through a Telegram bot.

Takes the alert dicts produced by Phase 4 (alert_engine.run_alert_engine)
and pushes them to a Telegram chat via the Bot API. Raw HTTP through
requests - the Bot API is a handful of GET/POST endpoints, so pulling in
a bot framework would buy nothing here (we never poll for commands).

Credentials live in secrets.env (gitignored via *.env), as:
    TELEGRAM_BOT_TOKEN=123456789:AA...
    TELEGRAM_CHAT_ID=987654321

Usage:
    python send_mes.py --chat-id    # discover your chat id after /start
    python send_mes.py --test       # send a hello message
    python send_mes.py --run        # run the alert engine and deliver
    python send_mes.py --preview    # print the last alert batch's digest, send nothing

Flags (with --run / --preview):
    --hybrid    classify news with the FinBERT+LLM hybrid path
    --no-ai     skip the per-ticker AI summary (e.g. when Ollama is off)
"""

import html
import os
import sys
import time

import requests

DEF_SECRETS_FILE = "secrets.env"
DEF_API_BASE = "https://api.telegram.org"

# (connect, read) - Telegram is normally fast; the read budget is mostly
# there so a hung connection can't stall a cron run forever
DEF_TIMEOUT = (10, 30)

# Bot API hard limit is 4096 UTF-16 code units per message; leave slack so
# a chunk boundary can't land us just over it
MAX_MESSAGE_CHARS = 3800

# a 429 hands back a retry_after; anything longer than this and we give up
# rather than block a scheduled run
MAX_RETRY_AFTER = 60
MAX_SEND_ATTEMPTS = 3

GREEN, RED, WHITE, WARN = "\U0001f7e2", "\U0001f534", "⚪", "⚠️"

VERDICT_EMOJI = {"bullish": GREEN, "bearish": RED}
LABEL_EMOJI = {"positive": GREEN, "neutral": WHITE, "negative": RED}

DEF_DIGEST_PROMPT_FILE = "data/digest_prompt.txt"

# plain-language reading of each price_data.py signal, plus which way it
# leans. oversold/overbought are tagged by the usual contrarian reading
# (oversold -> possible rebound), squeeze is direction-neutral.
SIGNAL_DESCRIPTIONS = {
    "golden_cross": (GREEN, "20-day average crossed above the 200-day (golden cross)"),
    "death_cross": (RED, "20-day average crossed below the 200-day (death cross)"),
    "rsi_oversold": (GREEN, "RSI under 30 - potentially oversold"),
    "rsi_overbought": (RED, "RSI over 70 - potentially overbought"),
    "macd_bullish": (GREEN, "MACD is above its signal line"),
    "macd_bearish": (RED, "MACD is below its signal line"),
    "macd_crossover_bull": (
        GREEN,
        "MACD line just crossed above zero - trend turning up",
    ),
    "macd_crossover_bear": (
        RED,
        "MACD line just crossed below zero - trend turning down",
    ),
    "boll_lower_band": (
        GREEN,
        "price closed below the lower Bollinger band - unusually low",
    ),
    "boll_upper_band": (
        RED,
        "price closed above the upper Bollinger band - unusually high",
    ),
    "squeeze": (
        WARN,
        "Bollinger bandwidth near a 1-year low - volatility squeeze, a big move may follow",
    ),
}


def load_secrets(path: str = DEF_SECRETS_FILE) -> dict:
    """Parse a KEY=VALUE file into a dict, without touching os.environ.

    Deliberately not python-dotenv: it isn't installed, and this keeps the
    script runnable from cron without a shell that sources anything.
    """
    secrets = {}
    if not os.path.exists(path):
        return secrets

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            secrets[key.strip()] = value.strip().strip("'\"")

    return secrets


# real env vars win over secrets.env, so a cron entry or a systemd unit can
# override the file without editing it
def _get_credential(name: str, secrets: dict | None = None) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    if secrets is None:
        secrets = load_secrets()
    return secrets.get(name)


def get_token(secrets: dict | None = None) -> str:
    token = _get_credential("TELEGRAM_BOT_TOKEN", secrets)
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set - add it to secrets.env "
            "(get one from @BotFather in Telegram)"
        )
    return token


def get_chat_id(secrets: dict | None = None) -> str:
    chat_id = _get_credential("TELEGRAM_CHAT_ID", secrets)
    if not chat_id:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID is not set - message your bot, then run "
            "`python send_mes.py --chat-id` to find it"
        )
    return chat_id


def _api_url(method: str, token: str) -> str:
    return f"{DEF_API_BASE}/bot{token}/{method}"


def _split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Split on line boundaries so an HTML tag never straddles two messages."""
    if len(text) <= limit:
        return [text]

    chunks, current = [], ""
    for line in text.split("\n"):
        # a single line longer than the limit is hard-cut; nothing we send
        # is built that way, but better truncated than rejected outright
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]

        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def send_message(
    text: str,
    token: str | None = None,
    chat_id: str | None = None,
    parse_mode: str = "HTML",
) -> bool:
    """Deliver one message, splitting it if it exceeds the Bot API limit."""
    secrets = load_secrets()
    token = token or get_token(secrets)
    chat_id = chat_id or get_chat_id(secrets)

    for chunk in _split_message(text):
        if not _send_chunk(chunk, token, chat_id, parse_mode):
            return False
    return True


def _send_chunk(chunk: str, token: str, chat_id: str, parse_mode: str) -> bool:
    payload = {
        "chat_id": chat_id,
        "text": chunk,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    for attempt in range(1, MAX_SEND_ATTEMPTS + 1):
        try:
            response = requests.post(
                _api_url("sendMessage", token), json=payload, timeout=DEF_TIMEOUT
            )
        except requests.RequestException as e:
            print(f"send_message: request failed ({e})")
            if attempt == MAX_SEND_ATTEMPTS:
                return False
            time.sleep(2 * attempt)
            continue

        # 429 is the only status worth waiting on - Telegram tells us how long
        if response.status_code == 429:
            retry_after = response.json().get("parameters", {}).get("retry_after", 1)
            if retry_after > MAX_RETRY_AFTER or attempt == MAX_SEND_ATTEMPTS:
                print(
                    f"send_message: rate limited, retry_after={retry_after}s, giving up"
                )
                return False
            print(f"send_message: rate limited, sleeping {retry_after}s")
            time.sleep(retry_after)
            continue

        if response.status_code != 200:
            print(f"send_message: HTTP {response.status_code} -> {response.text[:200]}")
            return False

        data = response.json()
        if not data.get("ok"):
            print(f"send_message: API error -> {data.get('description')}")
            return False

        return True

    return False


def discover_chat_id(token: str | None = None) -> list[dict]:
    """Read pending updates and report the chats that have messaged the bot.

    Only works for messages sent while no webhook is set and before the
    update queue expires (~24h), which is fine for one-time setup.
    """
    token = token or get_token()
    response = requests.get(_api_url("getUpdates", token), timeout=DEF_TIMEOUT)
    data = response.json()

    if not data.get("ok"):
        print(f"getUpdates failed -> {data.get('description')}")
        return []

    seen, chats = set(), []
    for update in data.get("result", []):
        message = update.get("message") or update.get("channel_post") or {}
        chat = message.get("chat")
        if not chat or chat["id"] in seen:
            continue
        seen.add(chat["id"])
        chats.append(chat)

    return chats


def _escape(value) -> str:
    return html.escape(str(value))


# --- briefing content ----------------------------------------------------


def technical_snapshot(ticker: str, hist_db: str = "data/hist.db") -> dict:
    """Current indicator values for a ticker, read back out of price_data's
    signal table (each signal row stores the indicator it was judged on)."""
    import sqlite3 as sql

    from alert_engine import _yf_ticker

    conn = sql.connect(hist_db)
    try:
        rows = dict(
            conn.execute(
                "SELECT signal_type, last_value FROM signal WHERE ticker = ?",
                (_yf_ticker(ticker),),
            ).fetchall()
        )
    finally:
        conn.close()

    return {
        "price": rows.get("boll_upper_band"),
        "rsi": rows.get("rsi_overbought"),
        "sma_spread": rows.get("golden_cross"),  # SMA20 - SMA200
        "macd_hist": rows.get("macd_bullish"),  # MACD - signal line
    }


def _fmt_num(value, signed: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _snapshot_line(snapshot: dict) -> str:
    return (
        f"Price {_fmt_num(snapshot.get('price'))} · RSI {_fmt_num(snapshot.get('rsi'))} · "
        f"SMA20-SMA200 {_fmt_num(snapshot.get('sma_spread'), signed=True)} · "
        f"MACD hist {_fmt_num(snapshot.get('macd_hist'), signed=True)}"
    )


def _describe_signal(signal_type: str) -> tuple[str, str]:
    return SIGNAL_DESCRIPTIONS.get(signal_type, (WARN, f"{signal_type} triggered"))


def _sentiment_context(
    ticker: str, alerts: list[dict], sentiment_row: dict | None
) -> dict:
    """Counts + links for a ticker: the current summarize_sentiment() row if
    we have one, else whatever the ticker's own sentiment alert carried
    (queued alerts replayed from an earlier run may be all we've got)."""
    if sentiment_row is not None:
        return {
            "counts": {
                k: sentiment_row.get(k, 0) for k in ("positive", "neutral", "negative")
            },
            "links": sentiment_row.get("links") or [],
        }
    for alert in alerts:
        if alert["alert_type"] == "sentiment":
            return {
                "counts": alert.get("counts", {}),
                "links": alert.get("links") or [],
            }
    return {"counts": {}, "links": []}


def _build_ai_input(
    ticker: str,
    company_name: str,
    alerts: list[dict],
    snapshot: dict,
    context: dict,
) -> str:
    lines = [f"Company: {company_name} (ticker {ticker})", ""]

    technical = [a for a in alerts if a["alert_type"] == "technical"]
    lines.append("Technical signals that just triggered:")
    if technical:
        for alert in technical:
            _, text = _describe_signal(alert.get("signal_type", ""))
            lines.append(f"- {text}")
    else:
        lines.append("- none")

    lines += ["", f"Indicator snapshot: {_snapshot_line(snapshot)}", ""]

    counts = context["counts"]
    lines.append(
        "News sentiment tally: {} positive, {} neutral, {} negative".format(
            counts.get("positive", 0),
            counts.get("neutral", 0),
            counts.get("negative", 0),
        )
    )
    lines.append("Articles:")
    for i, link in enumerate(context["links"], start=1):
        entry = f"{i}. [{link.get('label', 'unlabelled')}] {link.get('title') or link.get('url')}"
        if link.get("note"):
            entry += f" - analyst note: {link['note']}"
        lines.append(entry)
    if not context["links"]:
        lines.append("(no recent articles)")

    return "\n".join(lines)


def ai_summary(
    ticker: str,
    company_name: str,
    alerts: list[dict],
    snapshot: dict,
    context: dict,
    prompt_file: str = DEF_DIGEST_PROMPT_FILE,
) -> str | None:
    """Short LLM-written briefing tying the news and the signals together.
    Returns None rather than raising if Ollama is unreachable - a digest
    without the summary is still worth sending."""
    from sentiment_digest import DEF_ENDPOINT, DEF_MODEL, DEF_TEMPERATURE, DEF_TIMEOUT

    with open(prompt_file, encoding="utf-8") as handle:
        prompt = handle.read().format(company_name=company_name)

    data_sheet = _build_ai_input(ticker, company_name, alerts, snapshot, context)
    payload = {
        "model": DEF_MODEL,
        "messages": [{"role": "user", "content": f"{data_sheet}\n\n{prompt}"}],
        "stream": False,
        # same reason as sentiment_digest's calls: qwen's reasoning phase can
        # eat the whole output budget and leave content empty
        "think": False,
        "options": {"temperature": DEF_TEMPERATURE},
    }

    try:
        response = requests.post(DEF_ENDPOINT, json=payload, timeout=DEF_TIMEOUT)
        response.raise_for_status()
        text = response.json()["message"]["content"].strip()
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"ai_summary: {ticker} summary unavailable ({e})")
        return None

    return text or None


def format_ticker_briefing(
    ticker: str,
    alerts: list[dict],
    sentiment_row: dict | None = None,
    with_ai: bool = True,
    hist_db: str = "data/hist.db",
) -> str:
    """Full per-ticker section: signals in plain language, indicator
    snapshot, sentiment tally, AI summary, and every source link tagged
    with its own label."""
    from sentiment_digest import _company_name_for_ticker

    company_name = _company_name_for_ticker(ticker)
    snapshot = technical_snapshot(ticker, hist_db)
    context = _sentiment_context(ticker, alerts, sentiment_row)

    lines = [
        f"<b>{_escape(ticker)}</b> · {_escape(company_name)}",
        f"<code>{_escape(_snapshot_line(snapshot))}</code>",
    ]

    technical = [a for a in alerts if a["alert_type"] == "technical"]
    if technical:
        lines += ["", "\U0001f4c8 <b>Technical</b>"]
        for alert in technical:
            emoji, text = _describe_signal(alert.get("signal_type", ""))
            lines.append(f"{emoji} {_escape(text)}")

    counts = context["counts"]
    tally = "{} positive · {} neutral · {} negative".format(
        counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)
    )
    lines += ["", f"\U0001f4f0 <b>News sentiment</b>: {tally}"]
    for alert in alerts:
        if alert["alert_type"] == "sentiment" and alert.get("verdict"):
            verdict = alert["verdict"]
            lines.append(
                f"{VERDICT_EMOJI.get(verdict, WARN)} sentiment just turned <b>{_escape(verdict)}</b>"
            )

    # no articles -> nothing for the model to summarize; it would only spend
    # ~30s writing "there is no news", which the tally line already says
    if with_ai and context["links"]:
        summary = ai_summary(ticker, company_name, alerts, snapshot, context)
        if summary:
            lines += ["", f"\U0001f916 <i>{_escape(summary)}</i>"]
        else:
            lines += ["", "\U0001f916 <i>(AI summary unavailable this run)</i>"]

    if context["links"]:
        lines += ["", "<b>Sources</b>"]
        for link in context["links"]:
            url = link.get("url")
            if not url:
                continue
            emoji = LABEL_EMOJI.get(link.get("label"), WHITE)
            title = link.get("title") or url
            lines.append(f'{emoji} <a href="{_escape(url)}">{_escape(title)}</a>')

    return "\n".join(lines)


def format_digest(
    alerts: list[dict],
    sentiment_summary: list[dict] | None = None,
    with_ai: bool = True,
) -> str:
    """One briefing per alerted ticker, tickers in alphabetical order."""
    sentiment_by_ticker = {row["ticker"]: row for row in (sentiment_summary or [])}

    by_ticker: dict[str, list[dict]] = {}
    for alert in alerts:
        by_ticker.setdefault(alert["ticker"], []).append(alert)

    header = (
        f"<b>CrappyFinAssist</b> - {len(alerts)} new alert(s) "
        f"across {len(by_ticker)} ticker(s)"
    )
    sections = [
        format_ticker_briefing(
            ticker, by_ticker[ticker], sentiment_by_ticker.get(ticker), with_ai
        )
        for ticker in sorted(by_ticker)
    ]

    # a divider line between tickers; _split_message cuts on newlines, so a
    # long digest splits between (or inside) briefings, never mid-tag
    return header + "\n\n" + "\n\n━━━━━━━━━━\n\n".join(sections)


def send_alerts(
    alerts: list[dict],
    mark: bool = True,
    alert_db: str | None = None,
    sentiment_summary: list[dict] | None = None,
    with_ai: bool = True,
) -> bool:
    """Deliver a batch of alerts as one digest. No alerts -> no message.

    Only on a confirmed send are the rows flagged delivered, so a failure
    leaves them queued for the next run rather than losing them (the alert
    engine's dedup state has already moved on and won't re-fire them).
    Delivery is therefore at-least-once: a multi-chunk digest that dies
    halfway re-sends the whole digest next time, duplicating what did land.
    """
    if not alerts:
        print("send_alerts: nothing to send")
        return True

    if not send_message(format_digest(alerts, sentiment_summary, with_ai)):
        print(f"send_alerts: delivery failed, {len(alerts)} alert(s) stay queued")
        return False

    if mark:
        from alert_engine import DEF_ALERT_DB, mark_delivered

        mark_delivered([a.get("log_id") for a in alerts], alert_db or DEF_ALERT_DB)

    return True


def _cli() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "--run"

    if arg == "--chat-id":
        chats = discover_chat_id()
        if not chats:
            print(
                "No updates found. Open your bot in Telegram, press Start, send it "
                "a message, then run this again."
            )
            return 1
        for chat in chats:
            label = (
                chat.get("username") or chat.get("title") or chat.get("first_name", "")
            )
            print(f"chat_id={chat['id']}  type={chat.get('type')}  {label}")
        print("\nAdd the id you want to secrets.env as TELEGRAM_CHAT_ID=<id>")
        return 0

    if arg == "--test":
        ok = send_message("✅ <b>CrappyFinAssist</b> is wired up.")
        print("sent" if ok else "failed")
        return 0 if ok else 1

    use_hybrid = "--hybrid" in sys.argv
    with_ai = "--no-ai" not in sys.argv

    if arg == "--run":
        from alert_engine import pending_alerts, run_alert_engine
        from sentiment_digest import summarize_sentiment

        # read the backlog *before* the engine runs, so this run's own alerts
        # aren't counted twice - they arrive via run_alert_engine() instead
        queued = pending_alerts()
        if queued:
            print(f"{len(queued)} undelivered alert(s) from a previous run")

        # computed once and shared: the engine needs it to decide sentiment
        # alerts, the digest needs it for the full per-ticker picture
        summary = summarize_sentiment(use_hybrid=use_hybrid)
        alerts = run_alert_engine(sentiment_summary=summary)
        print(f"{len(alerts)} new alert(s)")

        ok = send_alerts(queued + alerts, sentiment_summary=summary, with_ai=with_ai)
        return 0 if ok else 1

    if arg == "--preview":
        # renders the most recent batch the engine fired, without sending,
        # marking anything delivered, or touching the engine's dedup state
        from alert_engine import latest_alert_batch
        from sentiment_digest import summarize_sentiment

        batch = latest_alert_batch()
        if not batch:
            print("No alerts have been fired yet - nothing to preview.")
            return 1
        summary = summarize_sentiment(use_hybrid=use_hybrid)
        print(format_digest(batch, summary, with_ai))
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(_cli())
