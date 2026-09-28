# stdlib
import json
import os
import random
import re
import sqlite3 as sql
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote, urlparse

# third-party
import requests
import trafilatura as tft
from ddgs import DDGS
from ddgs.exceptions import RatelimitException
from gnews import decode_google_news_url as dcd_ggl
from playwright.sync_api import (
    Error as PWError,
    TimeoutError as PWTimeoutError,
    sync_playwright,
)
from pygooglenews import GoogleNews
from transformers import pipeline

# local
from db_man import sql_to_df

blacklist = "VUAA"
DEF_TABLE = "company"
DEF_MAX_RESULTS = 5
DEF_MAX_AGE_HOURS = 48

# ft.com and marketwatch.com dropped 2026-09-16: both flooded site:
# searches with fund/ETF/index quote pages instead of news.
TRUSTED_SOURCES = {
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "cnbc.com",
    "apnews.com",
    "finance.yahoo.com",
    "fortune.com",
    "businessinsider.com",
    "economist.com",
    "news.google.com",
}

DEFAULT_DB = "data/news_hist.db"

CREATION_QUERY = """
    CREATE TABLE IF NOT EXISTS news (
        url TEXT,
        ticker TEXT,
        title TEXT,
        first_seen_at,
        PRIMARY KEY (url, ticker)
    )
"""

# persisted classification results, keyed by (url, ticker, method) rather
# than just (url, ticker) - the three classification paths (llm/finbert/
# hybrid) can legitimately disagree, so switching methods should mean a
# fresh classification, not silently reusing another method's answer
SENTIMENT_CACHE_QUERY = """
    CREATE TABLE IF NOT EXISTS sentiment_cache (
        url TEXT,
        ticker TEXT,
        method TEXT,
        label TEXT,
        result_json TEXT,
        classified_at TEXT,
        PRIMARY KEY (url, ticker, method)
    )
"""

DEF_MODEL = "qwen3.5:4b"
DEF_ENDPOINT = "http://192.168.88.241:11434/api/chat"
DEF_MODEL_PROMPT_FILE = "data/prompt.txt"
# escalation path (see hybrid_classify_article): asks the LLM for a plain
# opinion instead of strict JSON, then runs that opinion back through
# FinBERT for the structured label - sidesteps the LLM occasionally
# failing to produce valid JSON (see _parse_classification's fallback)
DEF_OPINION_PROMPT_FILE = "data/opinion_prompt.txt"
DEF_TEMPERATURE = 0.1
# (connect timeout, read timeout) - connecting to a LAN box should be
# near-instant, generation is what actually takes a while
DEF_TIMEOUT = (10, 180)
DEF_WARMUP_TIMEOUT = (10, 300)

MIN_EXTRACTED_CHARS = 200

# bot-wall / captcha pages extract as a few hundred chars of perfectly
# readable text, so they sail past MIN_EXTRACTED_CHARS and get classified
# (and summarized) as if they were the article. Real ones seen so far:
# Cloudflare's "security service" block (benzinga, thinkadvisor, kalkine)
# and TradingKey's "confirm you are human" puzzle. Only checked on short
# texts, so an actual article that mentions these phrases isn't dropped.
BLOCK_PAGE_MAX_CHARS = 1500
BLOCK_PAGE_MARKERS = (
    "security service to protect itself",
    "you were blocked",
    "confirm you are human",
    "verify you are human",
    "verifies that you are not a bot",
    "complete the security check",
    "checking your browser",
    "enable javascript and cookies",
    "access denied",
)


# cookie-consent notices can be *long* (CNBC's is ~7.5k chars) and stay in
# the DOM, hidden, after the dialog is dismissed. When the article body is
# missing - CNBC PRO paywall, for one - that notice is the biggest text
# block left and trafilatura extracts it. These are standard IAB consent
# framework phrases, so they're only checked at the very start of the text:
# no real article opens with them.
CONSENT_BOILERPLATE_MARKERS = (
    "store and/or access information on a device",
    "we and our vendors use cookies",
    "we and our partners use cookies",
    "we value your privacy",
)
CONSENT_CHECK_CHARS = 300


def _looks_like_block_page(text: str) -> bool:
    lowered = text.lower()
    if any(
        marker in lowered[:CONSENT_CHECK_CHARS]
        for marker in CONSENT_BOILERPLATE_MARKERS
    ):
        return True
    if len(text) > BLOCK_PAGE_MAX_CHARS:
        return False
    return any(marker in lowered for marker in BLOCK_PAGE_MARKERS)


# pacing between consecutive ddgs queries, to avoid firing a burst of
# back-to-back requests (the pattern most likely to get an IP temp-banned)
DDGS_MIN_DELAY = 0.5
DDGS_MAX_DELAY = 1.5
DDGS_MAX_RETRIES = 2
DDGS_BACKOFF_BASE = 5.0
# extra pacing between tickers in discover_news_for_tickers, on top of the
# per-source delay each fetch_recent_arts call already does internally
DDGS_TICKER_MIN_DELAY = 2.0
DDGS_TICKER_MAX_DELAY = 4.0


def elig_tickers(table_name=None):
    if table_name is None:
        print("No table supplied, going for DEF_TABLE")
        table_name = DEF_TABLE
    try:
        df = sql_to_df(table_name=table_name)
    except Exception as e:
        print(f"Maybe table_name is not gud -> {e}")
        return
    try:
        filtered_df = None
        if df is not None:
            filtered_df = df[["ticker", "name"]].copy()
    except Exception as e:
        print(f"filtered_df encountered an error -> {e}")
        return
    if filtered_df is None:
        return
    filtered_df = filtered_df[df["ticker"] != blacklist]  # type: ignore

    return filtered_df


def _parse_iso_date(date) -> datetime | None:
    if not date:
        return None
    try:
        return datetime.fromisoformat(date)
    except (TypeError, ValueError):
        return None


def _source_domain(url: str) -> str:
    domain = url.split("//", 1)[-1].split("/", 1)[0]
    return domain.removeprefix("www.")


# corporate-suffix / too-generic words to ignore when checking whether a
# search hit's title is actually about the company, vs. e.g. a fund or
# index quote page that just happens to match the site: search
GENERIC_NAME_WORDS = {
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "ltd",
    "ltd.",
    "plc",
    "se",
    "sa",
    "s.a.",
    "s.p.a.",
    "s.p.a",
    "group",
    "holdings",
    "technology",
    "technologies",
    "company",
}


def _title_matches_company(title: str, company_name: str) -> bool:
    title_lower = title.lower()
    words = [w.strip(".,") for w in company_name.split()]
    significant = [
        w for w in words if w.lower() not in GENERIC_NAME_WORDS and len(w) > 2
    ]
    if not significant:
        significant = words
    return any(w.lower() in title_lower for w in significant)


# wraps a single ddgs.news() call with exponential backoff on rate-limit
# errors specifically, rather than giving up on the first 429-style hit
def _ddgs_news_with_backoff(
    ddgs: DDGS, query: str, max_results: int, max_retries: int = DDGS_MAX_RETRIES
) -> list[dict]:
    delay = DDGS_BACKOFF_BASE
    for attempt in range(max_retries + 1):
        try:
            return list(ddgs.news(query, max_results=max_results))
        except RatelimitException:
            if attempt == max_retries:
                raise
            print(f"Rate limited on '{query}', backing off {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    return []  # unreachable, keeps type checkers happy


# a per-ticker request
# queries each domain in TRUSTED_SOURCES directly (via a site: restricted
# search) instead of a generic ddgs search, pools every hit, drops hits
# whose title doesn't actually reference the company (marketwatch/wsj in
# particular surface a lot of fund/index quote pages this way, which get
# indexed with today's date and would otherwise crowd out real news),
# sorts by date, and returns the most recent max_results across all of them.
# outputs a list of article in the format
# {title, url, source, date, snippet}
def fetch_recent_arts(
    company_name: str, max_results: int = DEF_MAX_RESULTS
) -> list[dict]:
    combined: list[dict] = []
    with DDGS() as ddgs:
        for source in TRUSTED_SOURCES:
            query = f'site:{source} "{company_name}"'
            try:
                results = _ddgs_news_with_backoff(ddgs, query, max_results)
            except RatelimitException as e:
                print(f"Search on {source} failed after retries (rate limited) -> {e}")
                continue
            except Exception as e:
                print(f"Search on {source} failed -> {e}")
                continue
            # ddgs's site: filter isn't strictly enforced - it can leak
            # results hosted on a different domain, so re-check here
            on_domain = [
                r for r in results if _source_domain(r["url"]).endswith(source)
            ]
            combined.extend(on_domain)
            time.sleep(random.uniform(DDGS_MIN_DELAY, DDGS_MAX_DELAY))

    combined = [
        art
        for art in combined
        if _title_matches_company(art.get("title", ""), company_name)
    ]

    combined.sort(
        key=lambda art: (
            _parse_iso_date(art.get("date", ""))
            or datetime.min.replace(tzinfo=timezone.utc)
        ),
        reverse=True,
    )
    return combined[:max_results]


# an alternative to duckduckgo's .news() method - broader coverage than
# the fixed TRUSTED_SOURCES list. Google News RSS entries link to an
# opaque news.google.com/rss/articles/... URL, decoded here via gnews's
# decode_google_news_url to the real publisher URL.
# {title, url, source, date, snippet}
def google_fetch_recent_arts(
    company_name: str, max_results: int = DEF_MAX_RESULTS
) -> list[dict]:
    gn = GoogleNews(lang="en", country="US")
    search_list = gn.search(company_name)

    results: list[dict] = []
    for entry in search_list["entries"]:
        if len(results) >= max_results:
            break
        time.sleep(random.uniform(DDGS_MIN_DELAY, DDGS_MAX_DELAY))

        results.append(
            {
                "title": entry.title,
                "url": dcd_ggl(entry.link),
                "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", entry.published_parsed),
            }
        )

    results.sort(
        key=lambda item: (
            _parse_iso_date(item.get("date", ""))
            or datetime.min.replace(tzinfo=timezone.utc)
        ),
        reverse=True,
    )
    return results


def relevance_filter(articles: list[dict], max_age_hours: int = DEF_MAX_AGE_HOURS):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    filtered = []
    for art in articles:
        art_date = _parse_iso_date(art.get("date", ""))
        if art_date is None or art_date < cutoff:
            continue
        art["trusted_source"] = _source_domain(art["url"]) in TRUSTED_SOURCES
        filtered.append(art)

    return filtered


def _create_news_db_table(db_name: str):
    conn = sql.connect(db_name)
    curr = conn.cursor()

    curr.execute(CREATION_QUERY)
    try:
        curr.execute("ALTER TABLE news ADD COLUMN title TEXT")
    except sql.OperationalError:
        pass  # column already exists on a pre-existing db

    conn.commit()
    conn.close()


def _ensure_sentiment_cache_table(conn: sql.Connection) -> None:
    conn.execute(SENTIMENT_CACHE_QUERY)
    conn.commit()


def _load_cached_result(
    conn: sql.Connection, url: str, ticker: str, method: str
) -> dict | None:
    row = conn.execute(
        "SELECT result_json FROM sentiment_cache WHERE url = ? AND ticker = ? AND method = ?",
        (url, ticker, method),
    ).fetchone()
    return json.loads(row[0]) if row else None


def _store_cached_result(
    conn: sql.Connection, url: str, ticker: str, method: str, result: dict
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO sentiment_cache (url, ticker, method, label, result_json, classified_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(url, ticker, method) DO UPDATE SET
            label = excluded.label,
            result_json = excluded.result_json,
            classified_at = excluded.classified_at
        """,
        (
            url,
            ticker,
            method,
            _normalize_sentiment_label(result),
            json.dumps(result),
            now,
        ),
    )
    conn.commit()


# remove already seen articles from the filtered list
def remove_seen(filtered: list[dict], conn: sql.Connection, ticker: str) -> list[dict]:
    curr = conn.cursor()

    new_article = []
    now = datetime.now(timezone.utc).isoformat()
    for art in filtered:
        url = art["url"]
        row = curr.execute(
            "SELECT 1 FROM news WHERE url=? AND ticker=?", (url, ticker)
        ).fetchone()

        if row is not None:
            continue
        curr.execute(
            "INSERT INTO news (url, ticker, title, first_seen_at) VALUES (?, ?, ?, ?)",
            (url, ticker, art.get("title"), now),
        )
        new_article.append(art)

    conn.commit()
    return new_article


SEARCH_BACKENDS = {
    "ddgs": fetch_recent_arts,
    "google": google_fetch_recent_arts,
}


# uses all the provided functions to retrieve a list of fresh news per
# ticker via the chosen backend (see SEARCH_BACKENDS):
# elig_tickers
# fetch_recent_arts / google_fetch_recent_arts
# relevance_filter
# remove_seen
def discover_news_for_tickers(
    db_name: str = DEFAULT_DB,
    max_age_hours: int = DEF_MAX_AGE_HOURS,
    max_results: int = DEF_MAX_RESULTS,
    backend: str = "ddgs",
) -> dict[str, list[dict]]:
    if backend not in SEARCH_BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}', pick one of {list(SEARCH_BACKENDS)}"
        )
    search = SEARCH_BACKENDS[backend]

    filtered_df = elig_tickers(DEF_TABLE)

    conn = sql.connect(db_name)
    _create_news_db_table(db_name)

    results: dict[str, list[dict]] = {}

    try:
        for ticker, name in filtered_df.itertuples(index=False):  # type: ignore
            raw = search(name, max_results=max_results)
            relevant = relevance_filter(raw, max_age_hours=max_age_hours)
            new = remove_seen(relevant, conn, ticker)

            results[ticker] = new
            # extra spacing between tickers, on top of what fetch_recent_arts
            # already does per-source, so a full run doesn't read as one
            # long unbroken burst of requests
            time.sleep(random.uniform(DDGS_TICKER_MIN_DELAY, DDGS_TICKER_MAX_DELAY))
    finally:
        conn.close()

    return results


def get_existing_articles(
    db_name: str = DEFAULT_DB,
    max_age_hours: int = DEF_MAX_AGE_HOURS,
    max_results: int = DEF_MAX_RESULTS,
) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}

    filtered_df = elig_tickers(DEF_TABLE)

    conn = sql.connect(db_name)
    conn.row_factory = sql.Row
    now = datetime.now(timezone.utc)
    try:
        for ticker, name in filtered_df.itertuples(index=False):  # type: ignore
            articles = conn.execute(
                "SELECT * FROM news WHERE ticker = ?", (ticker,)
            ).fetchall()
            for art in articles:
                art_date = _parse_iso_date(art["first_seen_at"])
                if art_date is None:
                    continue
                dif = now - art_date
                if dif > timedelta(hours=max_age_hours):
                    continue
                art_dict = dict(art)
                art_dict["trusted_source"] = (
                    _source_domain(art_dict["url"]) in TRUSTED_SOURCES
                )
                if ticker not in results:
                    results[ticker] = []
                results[ticker].append(art_dict)
                try:
                    if len(results[ticker]) >= max_results:
                        continue
                except KeyError:
                    continue
    finally:
        conn.close()
    return results


class BaseScraper:
    def extract_content(self, driver):
        raise NotImplementedError


class YahooScraper(BaseScraper):
    def extract_content(self, driver):
        # the "go to end" button isn't required to dismiss the consent
        # dialog, and chaining it before "accept all" meant a timeout on
        # this click alone (which happened often) skipped the accept
        # click entirely - confirmed by reproducing both clicks
        # independently. accept-all also needs a moment to settle before
        # the real page content is present in the DOM.
        driver.get_by_role("button", name="Acceptați tot").click(
            timeout=random.uniform(1.0, 2.0) * 1000
        )
        driver.wait_for_timeout(2000)


class Barrons_Binsider_Scraper(BaseScraper):
    def extract_content(self, driver):
        driver.locator('iframe[title="SP Consent Message"]').content_frame.get_by_role(
            "button", name="I Agree"
        ).click(timeout=random.uniform(1.0, 2.0) * 1000)


class FinTime_Scraper(BaseScraper):
    def extract_content(self, driver):
        return (
            driver.locator('iframe[title="SP Consent Message"]')
            .content_frame.get_by_role("button", name="Accept")
            .click(timeout=random.uniform(1.0, 2.0) * 1000)
        )


class GoogleScraper(BaseScraper):
    def extract_content(self, driver):
        driver.get_by_role("button", name="Accept all").click(
            timeout=random.uniform(3.0, 4.0) * 1000
        )


class NatGasIntelScraper(BaseScraper):
    def extract_content(self, driver):
        return driver.get_by_role("button", name="Reject Optional Cookies &").click(
            timeout=random.uniform(0.6, 1.0) * 1000
        )


class MarketScreenerScraper(BaseScraper):
    def extract_content(self, driver):
        return driver.get_by_role("button", name="Agree and close: Agree to our").click(
            timeout=random.uniform(0.6, 1.0) * 1000
        )


class CNBCScraper(BaseScraper):
    def extract_content(self, driver):
        return driver.get_by_role("button", name="Reject All").click(
            timeout=random.uniform(0.6, 1.0) * 1000
        )


def get_scraper_by_name(url: str) -> BaseScraper:
    match url:
        case u if "barrons.com" in u or "businessinsider.com" in u:
            return Barrons_Binsider_Scraper()
        case u if "finance.yahoo.com" in u:
            return YahooScraper()
        case u if "news.google.com" in u:
            return GoogleScraper()
        case u if "ft.com" in u:
            return FinTime_Scraper()
        case u if "marketscreener.com" in u:
            return MarketScreenerScraper()
        case u if "naturalgasintel.com" in u:
            return NatGasIntelScraper()
        case u if "cnbc.com" in u:
            return CNBCScraper()
        case _:
            raise ValueError(f"No custom scraper found matching URL -> {url}")


# how long a page may keep navigating (redirect chains, consent-click
# reloads) before we give up on it rather than wait forever
REDIRECT_MAX_WAIT = 15.0


def _wait_for_redirect(page, delay: float = 0.5, max_wait: float = REDIRECT_MAX_WAIT):
    """(content, title, url) once the page stops navigating. Raises
    PWTimeoutError if it's still redirecting after max_wait seconds."""
    deadline = time.monotonic() + max_wait
    while True:
        try:
            return page.content(), page.title(), page.url
        except PWError as e:
            if "changing the content" not in str(e):
                raise
            if time.monotonic() >= deadline:
                raise PWTimeoutError(
                    f"page still redirecting after {max_wait:.0f}s ({page.url})"
                ) from e
            # wait_for_timeout rather than time.sleep, so playwright keeps
            # processing the navigation events while we wait
            page.wait_for_timeout(delay * 1000)


# If there isn't any meaningful text extracted
# turn the url into something potentially useful
def extract_slug_text(url: str) -> str:
    # 1. Parse URL to isolate the path (removes query params like '?lang' and fragments)
    path = urlparse(url).path

    # 2. Get the last non-empty segment of the path
    segments = [s for s in path.split("/") if s]
    if not segments:
        return ""

    slug = segments[-1]

    # 3. Decode any URL encoding (e.g. %20)
    slug = unquote(slug)

    # 4. Remove common web page extensions (.html, .htm, etc.)
    slug = re.sub(r"\.(html?|php|asp|aspx)$", "", slug, flags=re.IGNORECASE)

    # 5. Replace dashes, underscores, and other non-alphanumeric chars with spaces
    cleaned = re.sub(r"[-_+]+", " ", slug)

    # 6. Normalize whitespace
    return " ".join(cleaned.split())


def _fetch_page(url: str | None, scraper: BaseScraper) -> str:
    if url is None:
        pass
    # HARNESS_HEADLESS=1 for unattended runs (cron has no display, and a
    # visible browser popping up every few hours isn't wanted anyway)
    headless = os.environ.get("HARNESS_HEADLESS") == "1"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            page.goto(url, timeout=10000)
        except PWError as e:
            # PWError covers timeouts too, plus things like net::ERR_HTTP2_
            # PROTOCOL_ERROR - one bad URL mustn't kill a whole run
            print(f"{url} failed to load -> {str(e).splitlines()[0]}")
            browser.close()
            return extract_slug_text(url)
        try:
            if isinstance(scraper, GoogleScraper):
                scraper.extract_content(driver=page)
                _, _, url = _wait_for_redirect(page)
                try:
                    scraper = get_scraper_by_name(_source_domain(url=url))
                except Exception:
                    pass
            try:
                scraper.extract_content(driver=page)
            except (PWTimeoutError, NotImplementedError, ValueError):
                pass  # no consent dialog on this site (maybe :) )
            # a consent click (or a late JS redirect) can leave the page
            # mid-navigation; plain page.content() raises then
            html, _, _ = _wait_for_redirect(page)
        except PWError as e:
            print(f"{url} never settled -> {str(e).splitlines()[0]}")
            browser.close()
            return extract_slug_text(url)
        browser.close()

    text = tft.extract(html)
    if text is None or len(text) < MIN_EXTRACTED_CHARS:
        text = extract_slug_text(url)  # type: ignore
    elif _looks_like_block_page(text):
        print(f"{url} gave a bot-check/consent page instead of the article")
        text = extract_slug_text(url)  # type: ignore
    return text[:8000]


def _parse_classification(reply: str) -> dict:
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {
        "sentiment": "neutral",
        "confidence": 0.0,
        "reason": "unparseable model output",
    }


def classify_art(
    article: dict,
    company_name: str,
    model: str = DEF_MODEL,
    endpoint: str = DEF_ENDPOINT,
    model_prompt_file: str = DEF_MODEL_PROMPT_FILE,
    temperature: float = DEF_TEMPERATURE,
    timeout: tuple[float, float] = DEF_TIMEOUT,
) -> dict:
    url = article.get("url")
    try:
        scraper = get_scraper_by_name(url)
    except Exception:
        scraper = BaseScraper()

    text = _fetch_page(url, scraper)
    with open(model_prompt_file, "r") as f:
        prompt = f.read().format(company_name=company_name)

    full_prompt = f"{text}\n\n{prompt}"

    # think=False: qwen3.5's reasoning mode can burn the whole output
    # token budget on its internal "thinking" trace before ever writing
    # to content, especially on longer articles - confirmed this leaves
    # content empty (done_reason "length") while thinking gets cut off
    # mid-sentence. Disabling it goes straight to a real answer.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }

    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    reply = resp.json()["message"]["content"]

    return _parse_classification(reply)


# same shape as classify_art, but returns the LLM's raw plain-text opinion
# instead of parsing it as JSON - see classify_art_via_opinion, which is
# the escalation path used instead of classify_art in hybrid_classify_article.
# article_text lets a caller that already fetched the page (e.g.
# hybrid_classify_article) pass it straight through, avoiding a second
# fetch and reusing whatever scraper-fallback the caller already resolved.
def get_llm_opinion(
    article: dict,
    company_name: str,
    model: str = DEF_MODEL,
    endpoint: str = DEF_ENDPOINT,
    model_prompt_file: str = DEF_OPINION_PROMPT_FILE,
    temperature: float = DEF_TEMPERATURE,
    timeout: tuple[float, float] = DEF_TIMEOUT,
    article_text: str | None = None,
) -> str:
    if article_text is None:
        url = article.get("url")
        try:
            scraper = get_scraper_by_name(_source_domain(url))
        except ValueError:
            scraper = BaseScraper()
        article_text = _fetch_page(url, scraper)

    with open(model_prompt_file, "r") as f:
        prompt = f.read().format(company_name=company_name)

    full_prompt = f"{article_text}\n\n{prompt}"

    # see classify_art's payload comment - think=False avoids the model
    # burning its output budget on internal reasoning before it ever
    # writes the actual opinion to content
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }

    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _warm_up_model(
    model: str = DEF_MODEL,
    endpoint: str = DEF_ENDPOINT,
    timeout: tuple[float, float] = DEF_WARMUP_TIMEOUT,
) -> None:
    """Force Ollama to load the model before the timed batch starts, so a
    cold load doesn't eat into classify_art's read timeout."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 1},
    }
    try:
        requests.post(endpoint, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        print(f"Model warm-up failed, continuing anyway -> {e}")


# type_of_inference: True -> classify_art (Ollama LLM),
# False -> basic_classifier_sentiment (local NLP model)
def aggregate_sentiment(
    ticker: str | None,
    model: str = DEF_MODEL,
    endpoint: str = DEF_ENDPOINT,
    type_of_inference: bool = False,
):
    if ticker is None:
        return
    hist = get_existing_articles().get(ticker, None)  # type: ignore
    if hist is None:
        print(f"No such ticker -> {ticker}")
        return

    method = "llm" if type_of_inference is True else "finbert"
    conn = sql.connect(DEFAULT_DB)
    _ensure_sentiment_cache_table(conn)

    results: list[dict] = []
    try:
        if type_of_inference is True:
            company_name = _company_name_for_ticker(ticker)
            warmed_up = False

            for art in hist:
                cached = _load_cached_result(conn, art["url"], ticker, method)
                if cached is not None:
                    results.append(cached)
                    continue

                if not warmed_up:
                    _warm_up_model(model=model, endpoint=endpoint)
                    warmed_up = True

                try:
                    result = classify_art(
                        art, company_name, model=model, endpoint=endpoint
                    )
                except requests.exceptions.RequestException as e:
                    print(f"Skipping '{art.get('title')}' ({art.get('url')}) -> {e}")
                    continue
                print(f"{art.get('url')} -> {result}")
                result["title"] = art.get("title")
                result["url"] = art.get("url")
                _store_cached_result(conn, art["url"], ticker, method, result)
                results.append(result)
        else:
            texts = []
            meta = []
            for art in hist:
                cached = _load_cached_result(conn, art["url"], ticker, method)
                if cached is not None:
                    results.append(cached)
                    continue
                url = art.get("url")
                domain = _source_domain(url)
                try:
                    scraper = get_scraper_by_name(domain)
                except ValueError:
                    scraper = BaseScraper()
                page = _fetch_page(url, scraper)
                texts.append(page)
                meta.append({"title": art.get("title"), "url": url})

            if texts:
                new_results = basic_classifier_sentiment(text=texts)
                for result, art_meta in zip(new_results, meta):
                    result["title"] = art_meta["title"]
                    result["url"] = art_meta["url"]
                    _store_cached_result(conn, art_meta["url"], ticker, method, result)
                    results.append(result)
    finally:
        conn.close()

    return results


def basic_classifier_sentiment(
    model: str = "ahmedrachid/FinancialBERT-Sentiment-Analysis", text: list[str] = []
) -> list[dict]:
    classifier = pipeline(
        "sentiment-analysis", model=model, truncation=True, max_length=512
    )

    results = classifier(text, batch_size=8)
    for idx, texts in enumerate(text):
        print(idx)
        print(texts)
    return results


# maps both classification schemas onto one common bucket set:
# classify_art (Ollama LLM) returns bullish/bearish/neutral (data/prompt.txt),
# basic_classifier_sentiment (FinBERT) returns positive/negative/neutral
# (confirmed via ahmedrachid/FinancialBERT-Sentiment-Analysis's id2label)
SENTIMENT_LABEL_MAP = {
    "bullish": "positive",
    "bearish": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "negative": "negative",
}


def _normalize_sentiment_label(result: dict) -> str:
    label = (result.get("sentiment") or result.get("label") or "").lower()
    return SENTIMENT_LABEL_MAP.get(label, "neutral")


# keyword-based check for whether an article discusses actual market
# dynamics at all, vs. being purely procedural/informational (e.g. local
# government infrastructure news that happens to mention large dollar
# figures). Validated against a small hand-reviewed sample: correctly
# flagged the one case that produced a false-positive FinBERT
# classification, and didn't trigger on any genuinely market-relevant
# article in that sample.
RELEVANCE_TERMS = [
    "shares",
    "share price",
    "stock",
    "%",
    "percent",
    "earnings",
    "eps",
    "guidance",
    "analyst",
    "upgrade",
    "downgrade",
    "rating",
    "trading",
    "revenue",
    "quarter",
    "buy point",
    "price target",
    "valuation",
    "forward earnings",
    "market cap",
]

# paragraphs shorter than this are treated as boilerplate/noise rather
# than real content when splitting an article for the disagreement check
PARAGRAPH_MIN_CHARS = 80
MIN_PARAGRAPHS_FOR_DISAGREEMENT_CHECK = 2


def _looks_market_relevant(text: str) -> bool:
    text_lower = text.lower()
    return any(term in text_lower for term in RELEVANCE_TERMS)


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n") if len(p.strip()) > PARAGRAPH_MIN_CHARS]


# escalation trigger: run FinBERT on the article's individual paragraphs
# and check whether the majority label disagrees with the whole-document
# label. Validated against a small hand-reviewed sample: caught both real
# misclassifications there (a mixed-narrative article and a magnitude-
# language false positive). A simpler keyword-based "hedging language"
# check (however/although/despite...) was tried first and rejected - it
# flagged nearly every article regardless of whether the whole-document
# call was actually right, so it wasn't a useful discriminator.
def _paragraph_majority_disagrees(text: str, doc_label: str) -> bool:
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) < MIN_PARAGRAPHS_FOR_DISAGREEMENT_CHECK:
        return False
    para_results = basic_classifier_sentiment(text=paragraphs)
    para_labels = [r["label"] for r in para_results]
    majority_label = max(set(para_labels), key=para_labels.count)
    return majority_label != doc_label


def _needs_escalation(text: str, doc_label: str) -> bool:
    if not _looks_market_relevant(text):
        return True
    return _paragraph_majority_disagrees(text, doc_label)


# escalation path used by hybrid_classify_article instead of classify_art:
# has the LLM write a plain-language opinion (get_llm_opinion) rather than
# trusting it to produce valid JSON directly, then classifies that opinion
# with FinBERT to get the structured label back
def classify_art_via_opinion(
    article: dict,
    company_name: str,
    model: str = DEF_MODEL,
    endpoint: str = DEF_ENDPOINT,
    opinion_prompt_file: str = DEF_OPINION_PROMPT_FILE,
    temperature: float = DEF_TEMPERATURE,
    timeout: tuple[float, float] = DEF_TIMEOUT,
    article_text: str | None = None,
) -> dict:
    opinion = get_llm_opinion(
        article,
        company_name,
        model=model,
        endpoint=endpoint,
        model_prompt_file=opinion_prompt_file,
        temperature=temperature,
        timeout=timeout,
        article_text=article_text,
    )
    result = basic_classifier_sentiment(text=[opinion])[0]
    result["opinion"] = opinion
    return result


# routes a single article to FinBERT (fast) or, if _needs_escalation trips,
# to the LLM-opinion + FinBERT chain instead. Returns a result dict
# enriched with title/url (same shape as aggregate_sentiment's results)
# plus an "escalated" flag so it's visible afterward which path was taken.
def hybrid_classify_article(
    article: dict,
    company_name: str,
    model: str = DEF_MODEL,
    endpoint: str = DEF_ENDPOINT,
) -> dict:
    url = article.get("url")
    domain = _source_domain(url)
    try:
        scraper = get_scraper_by_name(domain)
    except ValueError:
        scraper = BaseScraper()
    text = _fetch_page(url, scraper)

    baseline = basic_classifier_sentiment(text=[text])[0]

    if _needs_escalation(text, baseline["label"]):
        try:
            result = classify_art_via_opinion(
                article, company_name, model=model, endpoint=endpoint, article_text=text
            )
            result["escalated"] = True
        except requests.exceptions.RequestException as e:
            # Ollama went away mid-run: keep the FinBERT answer for now, and
            # flag it so aggregate_sentiment_hybrid doesn't cache it as the
            # hybrid verdict - next run retries the escalation
            print(f"escalation failed for {url}, keeping FinBERT baseline -> {e}")
            result = baseline
            result["escalated"] = False
            result["escalation_failed"] = True
    else:
        result = baseline
        result["escalated"] = False

    result["title"] = article.get("title")
    result["url"] = url
    return result


# hybrid counterpart to aggregate_sentiment: every article gets a fast
# FinBERT pass, and only the minority that trip an escalation signal (see
# _needs_escalation) get the slower LLM-opinion + FinBERT chain instead
def _company_name_for_ticker(ticker: str) -> str:
    filtered_df = elig_tickers(DEF_TABLE)
    if filtered_df is None:
        return ticker
    match = filtered_df[filtered_df["ticker"] == ticker]
    if match.empty:
        return ticker
    return match["name"].iloc[0]


def aggregate_sentiment_hybrid(
    ticker: str | None,
    model: str = DEF_MODEL,
    endpoint: str = DEF_ENDPOINT,
):
    if ticker is None:
        return None
    hist = get_existing_articles().get(ticker, None)  # type: ignore
    if hist is None:
        print(f"No such ticker -> {ticker}")
        return None

    company_name = _company_name_for_ticker(ticker)
    conn = sql.connect(DEFAULT_DB)
    _ensure_sentiment_cache_table(conn)

    results = []
    try:
        for art in hist:
            cached = _load_cached_result(conn, art["url"], ticker, "hybrid")
            if cached is not None:
                results.append(cached)
                continue

            result = hybrid_classify_article(
                art, company_name, model=model, endpoint=endpoint
            )
            tag = "escalated" if result["escalated"] else "finbert"
            print(f"[{tag}] {art.get('url')} -> {result}")
            if not result.get("escalation_failed"):
                _store_cached_result(conn, art["url"], ticker, "hybrid", result)
            results.append(result)
    finally:
        conn.close()

    return results


# final Phase 3 output: one dict per eligible ticker with how many of its
# sources came back positive/neutral/negative, plus the {title, url} of
# each source so they can be opened and read directly
def summarize_sentiment(
    model: str = DEF_MODEL,
    endpoint: str = DEF_ENDPOINT,
    type_of_inference: bool = False,
    use_hybrid: bool = False,
) -> list[dict]:
    filtered_df = elig_tickers(DEF_TABLE)
    tickers = list(filtered_df["ticker"]) if filtered_df is not None else []

    summary: list[dict] = []
    for ticker in tickers:
        if use_hybrid:
            results = (
                aggregate_sentiment_hybrid(ticker, model=model, endpoint=endpoint) or []
            )
        else:
            results = (
                aggregate_sentiment(
                    ticker,
                    model=model,
                    endpoint=endpoint,
                    type_of_inference=type_of_inference,
                )
                or []
            )

        counts = {"positive": 0, "neutral": 0, "negative": 0}
        links = []
        for result in results:
            label = _normalize_sentiment_label(result)
            counts[label] += 1
            # label + note travel with each link so the Phase 5 digest can
            # tag every source and feed the reasoning to its AI summary;
            # note is the LLM's opinion (hybrid) or reason (llm), None for
            # plain FinBERT results
            links.append(
                {
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "label": label,
                    "note": result.get("opinion") or result.get("reason"),
                }
            )

        summary.append(
            {
                "ticker": ticker,
                "positive": counts["positive"],
                "neutral": counts["neutral"],
                "negative": counts["negative"],
                "links": links,
            }
        )

    return summary


if __name__ == "__main__":
    # conn = sql.connect(DEFAULT_DB)
    # conn.execute("DELETE FROM news")
    # conn.commit()
    # conn.close()
    # found = discover_news_for_tickers(backend="google", max_age_hours=200)

    # print("News just discovered...")
    # for ticker, articles in found.items():
    #    print(f"{ticker}: {len(articles)} new article(s)")
    #    for article in articles:
    #        trust_tag = "trusted" if article["trusted_source"] else "unverified"
    #        print(f"\t[{trust_tag}] {article['title']} - {article['url']}")

    hist = get_existing_articles()
    print("News from history")
    for ticker, articles in hist.items():
        print(f"{ticker}: {len(articles)} new article(s)")
        for article in articles:
            trust_tag = "trusted" if article["trusted_source"] else "unverified"
            print(f"\t[{trust_tag}] {article['title']} - {article['url']}")

    results = aggregate_sentiment("TOTB")
    for resp in results:
        print(resp)
