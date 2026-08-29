"""
scraper.py
Scraper for stopfals.md — the Republic of Moldova's fact-checking portal.
Falls back gracefully if the site is unreachable.

Usage:
    python src/scraper.py --pages 10 --output data/raw/stopfals.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Iterator, List, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

BASE_URL = "https://stopfals.md"
ARTICLE_LIST_PATH = "/ro/articles"
REQUEST_DELAY = 1.5  # seconds between requests — be polite


@dataclass
class Article:
    url: str
    claim_text: str
    verdict: str
    justification: str
    date: str
    tags: List[str] = field(default_factory=list)
    evidence_text: str = ""


# ---------------------------------------------------------------------------
# HTTP helpers — use requests with a well-identified User-Agent
# ---------------------------------------------------------------------------

def _get_session():
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as exc:
        raise ImportError("Install requests: pip install requests") from exc

    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "StopFalsResearchScraper/1.0 "
                "(academic dissertation; contact: research@example.com)"
            ),
            "Accept-Language": "ro,en;q=0.9",
        }
    )
    return session


def _fetch(session, url: str) -> Optional[str]:
    """Fetch a URL and return the response text, or None on failure."""
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_article_links(html: str, base: str = BASE_URL) -> List[str]:
    """Extract article URLs from a listing page."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4") from exc

    soup = BeautifulSoup(html, "html.parser")
    links = []
    # Typical stopfals.md article card anchor pattern
    for a in soup.select("a[href*='/ro/article/'], a[href*='/article/']"):
        href = a.get("href", "")
        full = urljoin(base, href)
        if full not in links:
            links.append(full)
    return links


def _parse_article(html: str, url: str) -> Optional[Article]:
    """Parse an individual fact-checking article page."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4") from exc

    soup = BeautifulSoup(html, "html.parser")

    # --- Claim text ---
    claim_el = (
        soup.select_one(".claim-text")
        or soup.select_one("blockquote")
        or soup.select_one("h2")
    )
    claim_text = claim_el.get_text(strip=True) if claim_el else ""

    # --- Verdict ---
    verdict_el = soup.select_one(".verdict") or soup.select_one(".rating")
    verdict = verdict_el.get_text(strip=True) if verdict_el else ""

    # --- Justification (article body) ---
    body_el = soup.select_one(".article-body") or soup.select_one("article")
    paragraphs = body_el.find_all("p") if body_el else []
    justification = " ".join(p.get_text(strip=True) for p in paragraphs)

    # --- Evidence links / sources ---
    sources = []
    if body_el:
        for a in body_el.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http"):
                sources.append(href)
    evidence_text = "; ".join(sources[:10])

    # --- Date ---
    date_el = soup.select_one("time") or soup.select_one(".date")
    date = date_el.get("datetime", date_el.get_text(strip=True)) if date_el else ""

    # --- Tags ---
    tags = [t.get_text(strip=True) for t in soup.select(".tag, .tags a")]

    if not claim_text and not justification:
        return None

    return Article(
        url=url,
        claim_text=claim_text,
        verdict=verdict,
        justification=justification,
        date=date,
        tags=tags,
        evidence_text=evidence_text,
    )


# ---------------------------------------------------------------------------
# Main scraping routine
# ---------------------------------------------------------------------------

def scrape_stopfals(max_pages: int = 10) -> Iterator[Article]:
    """
    Generator that yields Article objects from stopfals.md.
    Iterates over listing pages up to `max_pages`.
    """
    session = _get_session()

    for page_num in range(1, max_pages + 1):
        list_url = f"{BASE_URL}{ARTICLE_LIST_PATH}?page={page_num}"
        logger.info("Fetching listing page %d: %s", page_num, list_url)
        list_html = _fetch(session, list_url)
        if not list_html:
            logger.warning("Could not fetch listing page %d — stopping.", page_num)
            break

        article_links = _parse_article_links(list_html)
        if not article_links:
            logger.info("No article links found on page %d — done.", page_num)
            break

        for link in article_links:
            time.sleep(REQUEST_DELAY)
            art_html = _fetch(session, link)
            if not art_html:
                continue
            article = _parse_article(art_html, link)
            if article:
                yield article

        time.sleep(REQUEST_DELAY)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape fact-checking articles from stopfals.md"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=10,
        help="Maximum number of listing pages to scrape (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/stopfals.jsonl",
        help="Output JSONL file path (default: data/raw/stopfals.jsonl)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for article in scrape_stopfals(max_pages=args.pages):
            f.write(json.dumps(asdict(article), ensure_ascii=False) + "\n")
            count += 1
            if count % 10 == 0:
                logger.info("Scraped %d articles so far …", count)

    logger.info("Done. Total articles saved: %d → %s", count, args.output)


if __name__ == "__main__":
    main()
