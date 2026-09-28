import sentiment_digest as sd
from transformers import pipeline
import re
from urllib.parse import unquote, urlparse


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


results = sd.google_fetch_recent_arts("Total Energies SE", 25)

classifier = pipeline(
    "sentiment-analysis",
    model="ahmedrachid/FinancialBERT-Sentiment-Analysis",
    truncation=True,
    max_length=512,
)

for idx, art_dict in enumerate(results):
    print(f"{idx}\n{art_dict['title']}\n{art_dict['url']}\n{art_dict['date']}")
    try:
        scraper = sd.get_scraper_by_name(sd._source_domain(art_dict["url"]))
    except Exception as e:
        print(e)
        scraper = sd.BaseScraper()
    page = sd._fetch_page(art_dict["url"], scraper)
    if "Could not extract readable content" in page:
        page = extract_slug_text(art_dict["url"])

    print(page)

    print(classifier(page))
