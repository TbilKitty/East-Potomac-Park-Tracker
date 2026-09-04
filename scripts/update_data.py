import os
import json
import time
import html as html_module
from datetime import datetime, timezone

import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DOCKET_ID = 72278895  # DC Preservation League v. Department of Interior, 1:26-cv-00477
API_TOKEN = os.environ.get("COURTLISTENER_TOKEN", "")
headers = {"Authorization": f"Token {API_TOKEN}"} if API_TOKEN else {}


def get_docket_entries(docket_id):
    entries = []
    url = "https://www.courtlistener.com/api/rest/v4/docket-entries/"
    params = {"docket": docket_id, "order_by": "-date_filed"}
    deadline = time.monotonic() + 180
    rate_limit_retries = 0
    pages = 0
    while url:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or pages >= 20:
            raise requests.exceptions.Timeout("CourtListener fetch limit reached; use saved docket data.")
        print(f"CourtListener: fetching page {pages + 1}...", flush=True)
        resp = requests.get(url, params=params, headers=headers, timeout=min(30, remaining))
        if resp.status_code == 429:
            rate_limit_retries += 1
            if rate_limit_retries > 3:
                resp.raise_for_status()
            delay = 15 * rate_limit_retries
            if time.monotonic() + delay >= deadline:
                raise requests.exceptions.Timeout("CourtListener rate limit exceeded fetch budget.")
            print(f"CourtListener rate limited; retry {rate_limit_retries}/3 in {delay}s.", flush=True)
            time.sleep(delay)
            continue
        resp.raise_for_status()
        data = resp.json()
        entries.extend(data["results"])
        pages += 1
        url = data.get("next")
        params = {}
        if url:
            remaining = deadline - time.monotonic()
            if remaining <= 13:
                raise requests.exceptions.Timeout("CourtListener  budget exhausted.")
            time.sleep(13)
    return entries


def gdelt_search(query, start, end, maxrecords=250, retries=3):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query, "mode": "artlist", "format": "json",
        "maxrecords": maxrecords, "startdatetime": start,
        "enddatetime": end, "sort": "datedesc",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("articles", [])
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                return []
            time.sleep(10)
    return []


def get_federal_register_notices(term, agency):
    url = "https://www.federalregister.gov/api/v1/documents.json"
    params = {
        "conditions[term]": term,
        "conditions[agencies][]": agency,
        "order": "newest",
        "per_page": 10,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"Federal Register  failed: {e}")
        return []


HEARING_KEYWORDS = ["hearing", "conference", "oral argument", "status conference", "scheduling order"]

def flag_hearing_entries(df_docket):
    if df_docket.empty:
        return df_docket
    mask = df_docket["description"].str.lower().str.contains("|".join(HEARING_KEYWORDS), na=False)
    return df_docket[mask]


def send_email_update(subject, body):
    api_key = os.environ.get("BUTTONDOWN_API_KEY", "")
    if not api_key:
        print("No BUTTONDOWN_API_KEY set, skipping email send.")
        return
    url = "https://api.buttondown.com/v1/emails"
    headers = {
        "Authorization": f"Token {api_key}",
        "X-Buttondown-Live-Dangerously": "true",
        "Content-Type": "application/json",
    }
    payload = {"subject": subject, "body": body, "status": "about_to_send"}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        print("Email update sent.")
    except requests.exceptions.RequestException as e:
        print(f"Email send failed: {e}")


trends_html = f"""
<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/4179_RC01/embed_loader.js"></script>
<div class="trends-widget" style="font-family:Arial,sans-serif;"></div>
<script type="text/javascript">
  trends.embed.renderExploreWidget(
    "TIMESERIES",
    {{"comparisonItem":[{{"keyword":"East Potomac Park","geo":"US","time":"today 3-m"}}],"category":0,"property":""}},
    {{"exploreQuery":"date=today%203-m&geo=US&q=East%20Potomac%20Park&hl=en","guestPath":"https://trends.google.com:443/trends/embed/"}}
  );
</script>
"""


import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup

ARTICLE_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EastPotomacParkTracker/1.0; +https://github.com)"
}


JUNK_CONTAINER_SELECTORS = ["nav", "footer", "aside", "header"]
JUNK_CLASS_KEYWORDS = [
    "related", "recommend", "trending", "promo", "newsletter",
    "social", "comment", "sidebar", "advert", "subscribe", "share",
]


def fetch_article_text(url, timeout=15, max_chars=4000):
    try:
        resp = requests.get(url, headers=ARTICLE_FETCH_HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip obvious non-article chrome before we go looking for text.
        for tag_name in JUNK_CONTAINER_SELECTORS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        for tag in soup.find_all(class_=True):
            classes = " ".join(tag.get("class", [])).lower()
            if any(keyword in classes for keyword in JUNK_CLASS_KEYWORDS):
                tag.decompose()

        # Prefer a real <article> tag or a common main-content container if one exists.
        container = soup.find("article")
        if container is None:
            container = soup.find(attrs={"class": re.compile(r"(article|story|entry)[-_]?(body|content)", re.I)})
        if container is None:
            container = soup.find("main")
        if container is None:
            container = soup  # fall back to whatever's left after stripping junk

        paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
        text = " ".join(p for p in paragraphs if len(p) > 40)
        return text[:max_chars] if text else None
    except Exception:
        return None


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def summarize_text(text, num_sentences=2):
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(sentences)
        scores = tfidf.sum(axis=1).A1
        top_idx = sorted(scores.argsort()[-num_sentences:])  # keep original order
        return " ".join(sentences[i] for i in top_idx)
    except ValueError:
        return " ".join(sentences[:num_sentences])


ARTICLE_STORE_PATH = "data/articles_seen.json"

# Recovered from the site history at commit e5d84e8cc3d3bec614ebe5ca331449057d443734.
RECOVERED_ARTICLES = [{'url': 'https://wtop.com/dc/2026/06/lawmakers-say-white-house-demolition-debris-at-east-potomac-park-poses-health-risk/',
  'title': 'Lawmakers say White House demolition debris at East Potomac Park '
           'poses health risk',
  'domain': 'wtop.com',
  'seendate': '2026-06-27 00:15:00+00:00'},
 {'url': 'https://www.yahoo.com/news/politics/articles/democrats-demand-trump-remove-east-164805508.html',
  'title': 'Democrats demand Trump remove East Wing debris  recklessly  dumped '
           'at East Potomac Park',
  'domain': 'yahoo.com',
  'seendate': '2026-06-26 20:15:00+00:00'}]


def load_article_store(path=ARTICLE_STORE_PATH):
    """Every article ever found, keyed by URL, with its cached scraped text."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                items = json.load(f)
            return {item["url"]: item for item in items}
        except (json.JSONDecodeError, KeyError, OSError) as e:
            raise RuntimeError("Article archive could not be read; refusing to overwrite it.") from e
    return {}


def save_article_store(store, path=ARTICLE_STORE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Don't persist internal sort helpers.
    clean = [{k: v for k, v in item.items() if not k.startswith("_")} for item in store.values()]
    with open(path + ".tmp", "w") as f:
        json.dump(clean, f, indent=2, default=str)
    os.replace(path + ".tmp", path)


def merge_new_articles_into_store(store, df_media):
    """Fetch + cache text only for URLs we haven't seen before."""
    new_count = 0
    if df_media.empty:
        return new_count
    for _, row in df_media.iterrows():
        url = row.get("url")
        if not url or url in store:
            continue
        text = fetch_article_text(url)
        time.sleep(1)  # be polite to the sites we're fetching from
        seendate = row.get("seendate")
        seendate_str = seendate.isoformat() if hasattr(seendate, "isoformat") else str(seendate)
        store[url] = {
            "url": url,
            "title": row.get("title", "(untitled)"),
            "domain": row.get("domain", ""),
            "seendate": seendate_str,
            "fetched_text": text if text else row.get("title", ""),
        }
        new_count += 1
    return new_count


def rank_stored_articles_by_novelty(store):
    """Recompute novelty over the FULL accumulated history so scores stay
    comparable across old and newly-added articles, then sort for display
    (most novel first). Order of computation is always chronological by
    seendate so results are stable regardless of when the script runs."""
    if not store:
        return []

    items = list(store.values())
    for item in items:
        item["_seendate_dt"] = pd.to_datetime(item["seendate"], errors="coerce", utc=True)
    # Keep undated records too; do not silently remove them from the list.
    items.sort(key=lambda x: (pd.isna(x["_seendate_dt"]),
                             x["_seendate_dt"] if pd.notna(x["_seendate_dt"]) else pd.Timestamp.max.tz_localize("UTC"),
                             x["url"]))

    texts = [item.get("fetched_text") or item.get("title", "") for item in items]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # No usable vocabulary: retain every article without inventing scores.
        for item in items:
            item["novelty"] = None
            item["summary"] = item.get("title", "")
            item.pop("_seendate_dt", None)
        return items

    seen_vector = None
    for i, item in enumerate(items):
        current = tfidf_matrix[i]
        if seen_vector is None:
            novelty = 1.0
        else:
            sim = cosine_similarity(current, seen_vector)[0][0]
            novelty = max(0.0, 1.0 - sim)
        seen_vector = current if seen_vector is None else seen_vector + current
        item["novelty"] = novelty
        item["summary"] = summarize_text(texts[i])

    items.sort(key=lambda r: r["novelty"], reverse=True)
    for item in items:
        item.pop("_seendate_dt", None)
    return items


PHOTO_URL = "https://www.nps.gov/npgallery/GetAsset/36B74D30-DE82-4B2E-8A20-F83F69B55B39"


def ensure_header_photo():
    """Keep the NPS photo with the generated site so Pages can serve it."""
    path = "park-photo.jpg"
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    try:
        response = requests.get(PHOTO_URL, timeout=30)
        response.raise_for_status()
        if not response.content.startswith(b"\xff\xd8\xff"):
            raise ValueError("NPS response was not a JPEG")
        with open(path + ".tmp", "wb") as photo:
            photo.write(response.content)
        os.replace(path + ".tmp", path)
        return path
    except (requests.RequestException, OSError, ValueError) as exc:
        print(f"Header photo download failed; using NPS source: {exc}")
        return PHOTO_URL

def main():
    os.makedirs("data", exist_ok=True)
    now = datetime.now(timezone.utc)
    photo_src = ensure_header_photo()
    # --- Docket ---
    df_docket = pd.DataFrame(columns=["date_filed", "entry_number", "description"])
    try:
        entries = get_docket_entries(DOCKET_ID)
        if entries:
            df_docket = pd.DataFrame(entries)
            for col in ["date_filed", "entry_number", "description"]:
                if col not in df_docket.columns:
                    df_docket[col] = None
            df_docket = df_docket[["date_filed", "entry_number", "description"]].sort_values(
                "date_filed", ascending=False)
    except (requests.exceptions.RequestException, KeyError) as e:
        print(f"CourtListener fetch/parse failed, keeping previous data if any: {e}")
        if os.path.exists("data/east_potomac_docket.csv"):
            df_docket = pd.read_csv("data/east_potomac_docket.csv")
    df_docket.to_csv("data/east_potomac_docket.csv", index=False)
#below line was left from prior file
    print("Fetching public notices...", flush=True)
    # --- Upcoming hearings / notices ---
    fr_notices = get_federal_register_notices("East Potomac", "national-park-service")
    fr_notices += get_federal_register_notices("East Potomac", "interior-department")
    hearing_entries = flag_hearing_entries(df_docket)

    article_store = load_article_store()
    for article in RECOVERED_ARTICLES:
        if article["url"] not in article_store:
            article_store[article["url"]] = dict(article, fetched_text=article["title"])
    media_path = "data/east_potomac_media.csv"
    if os.path.exists(media_path):
        try:
            previous_media = pd.read_csv(media_path)
        except pd.errors.EmptyDataError:
            previous_media = pd.DataFrame()
        merge_new_articles_into_store(article_store, previous_media)
    save_article_store(article_store)

    print("Fetching news articles...", flush=True)
    # --- Media (rolling 90-day window, GDELT's actual coverage range) ---
    start = (now - pd.Timedelta(days=90)).strftime("%Y%m%d%H%M%S")
    end = now.strftime("%Y%m%d%H%M%S")
    df_media = pd.DataFrame()
    try:
        articles = gdelt_search('"East Potomac Park"', start, end)
        if articles:
            df_media = pd.DataFrame(articles)
            df_media["seendate"] = pd.to_datetime(df_media["seendate"])
            # Belt-and-suspenders: GDELT's query filtering isn't fully reliable,
            # so double-check locally that the park is actually mentioned.
            mask = df_media["title"].str.contains("east potomac", case=False, na=False)
            df_media = df_media[mask]
    except Exception as e:
        print(f"GDELT fetch failed, keeping previous data if any: {e}")
        if os.path.exists("data/east_potomac_media.csv"):
            df_media = pd.read_csv("data/east_potomac_media.csv")
            if not df_media.empty:
                df_media["seendate"] = pd.to_datetime(df_media["seendate"])
    # An empty result or failed search must never erase the saved snapshot.
    if not df_media.empty:
        df_media.to_csv(media_path + ".tmp", index=False)
        os.replace(media_path + ".tmp", media_path)

    # --- Article store: merge in only genuinely new URLs, then re-rank the
    # full accumulated history so nothing that's already been featured
    # disappears, and novelty scores stay comparable across runs. ---
    new_count = merge_new_articles_into_store(article_store, df_media)
    print(f"{new_count} new article(s) this run; {len(article_store)} total in store.", flush=True)

    print("Re-ranking full article history by novelty...", flush=True)
    ranked_articles = rank_stored_articles_by_novelty(article_store)
    save_article_store(article_store)

    print("Fetching Google Trends search interest...", flush=True)
    trend_df = fetch_search_interest()
    if trend_df is not None and not trend_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(trend_df.index, trend_df["East Potomac Park"], marker="o", color="#2C5F4F", linewidth=2)
        ax.fill_between(trend_df.index, trend_df["East Potomac Park"], color="#2C5F4F", alpha=0.08)
        ax.set_title("Search Interest \u2014 \u201cEast Potomac Park\u201d (Google Trends, US)")
        ax.set_ylabel("Relative interest (0\u2013100)")
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.spines[["top", "right"]].set_visible(False)
        fig.autofmt_xdate(rotation=40)
        plt.tight_layout()
        plt.savefig("search_interest.png", dpi=150)
        plt.close()
        trends_available = True
    else:
        trends_available = os.path.exists("search_interest.png")
        print("Using previously saved trends chart (today's fetch failed or returned nothing).", flush=True)

    print("Rendering the updated page...", flush=True)
    # --- Render index.html ---
    updated = now.strftime("%B %d, %Y at %H:%M UTC")

    docket_rows = ""
    for _, row in df_docket.head(30).iterrows():
        docket_rows += (
            f"<tr><td>{html_module.escape(str(row['date_filed']))}</td>"
            f"<td>{html_module.escape(str(row['entry_number']))}</td>"
            f"<td>{html_module.escape(str(row['description']))}</td></tr>\n"
        )
    if not docket_rows:
        docket_rows = "<tr><td colspan='3'>No docket entries retrieved yet.</td></tr>"

    hearing_rows = ""
    for _, row in hearing_entries.iterrows():
        hearing_rows += f"<li><strong>{html_module.escape(str(row['date_filed']))}</strong> \u2014 {html_module.escape(str(row['description']))}</li>\n"

    notice_rows = ""
    for doc in fr_notices:
        title = html_module.escape(doc.get("title", ""))
        pub_date = html_module.escape(doc.get("publication_date", ""))
        html_url = html_module.escape(doc.get("html_url", "#"))
        notice_rows += f'<li><strong>{pub_date}</strong> \u2014 <a href="{html_url}" target="_blank" rel="noopener">{title}</a></li>\n'

    if not hearing_rows and not notice_rows:
        hearings_section = "<p>No public hearings or notices found in the current data. This section updates daily.</p>"
    else:
        hearings_section = ""
        if hearing_rows:
            hearings_section += (
                f'<details><summary>Court Hearings &amp; Conferences ({len(hearing_entries)})</summary>'
                f'<ul>{hearing_rows}</ul></details>'
            )
        if notice_rows:
            hearings_section += (
                f'<details><summary>Federal Register Notices ({len(fr_notices)})</summary>'
                f'<ul>{notice_rows}</ul></details>'
            )

    # --- Compare against last run to see if anything is genuinely new ---
    current_items = set()
    for _, row in hearing_entries.iterrows():
        current_items.add(f"hearing::{row['date_filed']}::{row['description']}")
    for doc in fr_notices:
        current_items.add(f"notice::{doc.get('publication_date','')}::{doc.get('title','')}")

    prev_path = "data/previous_hearings.json"
    previous_items = set()
    had_previous = os.path.exists(prev_path)
    if had_previous:
        with open(prev_path) as f:
            previous_items = set(json.load(f))

    new_items = current_items - previous_items
    if had_previous and new_items:
        lines = []
        for item in new_items:
            kind, date, desc = item.split("::", 2)
            label = "Court hearing/conference" if kind == "hearing" else "Federal Register notice"
            lines.append(f"- **{label}** ({date}): {desc}")
        body = (
            "New hearing or public notice activity for the East Potomac Park case:\n\n"
            + "\n".join(lines)
            + f"\n\nFull tracker: https://{os.environ.get('GITHUB_REPOSITORY_OWNER', 'yourusername')}.github.io/"
        )
        send_email_update("East Potomac Park: New hearing/notice update", body)

    os.makedirs("data", exist_ok=True)
    with open(prev_path, "w") as f:
        json.dump(sorted(current_items), f)

    def render_article_div(a):
        pct = round(a["novelty"] * 100) if a["novelty"] is not None else None
        score_label = f"{pct}% novelty score" if pct is not None else "Not scored"
        seendate_dt = pd.to_datetime(a["seendate"], errors="coerce")
        date_str = seendate_dt.strftime("%b %d, %Y") if pd.notna(seendate_dt) else str(a["seendate"])
        safe_title = html_module.escape(str(a["title"]))
        safe_domain = html_module.escape(str(a["domain"]))
        safe_summary = html_module.escape(str(a["summary"]))
        safe_url = html_module.escape(str(a["url"]))
        return f"""
        <div style="border-bottom:1px solid #D8D3C7; padding:16px 0;">
          <div style="display:flex; justify-content:space-between; align-items:baseline; gap:12px;">
            <a href="{safe_url}" target="_blank" rel="noopener" style="font-weight:bold; color:#1A1A1A; text-decoration:none;">{safe_title}</a>
            <span style="font-family:Arial,sans-serif; font-size:0.78rem; font-weight:bold; color:#2C5F4F; white-space:nowrap;">{score_label}</span>
          </div>
          <div style="font-family:Arial,sans-serif; font-size:0.78rem; color:#4A4A4A; margin:2px 0 8px;">{safe_domain} &middot; {date_str}</div>
          <div style="font-size:0.92rem; color:#333;">{safe_summary}</div>
        </div>
        """

    TOP_N_VISIBLE = 3
    top_articles = ranked_articles[:TOP_N_VISIBLE]
    rest_articles = ranked_articles[TOP_N_VISIBLE:]

    article_rows = "".join(render_article_div(a) for a in top_articles)
    if rest_articles:
        rest_rows = "".join(render_article_div(a) for a in rest_articles)
        article_rows += (
            f'<details class="more-articles"><summary>Show {len(rest_articles)} more article'
            f'{"s" if len(rest_articles) != 1 else ""}</summary>{rest_rows}</details>'
        )
    if not article_rows:
        article_rows = "<p>No articles available to rank this run.</p>"

    trends_html = (
        '<img src="search_interest.png" alt="Google Trends search interest for East Potomac Park" style="width:100%; border-radius:8px;">'
        if trends_available else
        '<p>Search interest data not available this run.</p>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>East Potomac Park Case Tracker</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 900px; margin: 0 auto; padding: 40px 20px; background: #FAF7F1; color: #1A1A1A; }}
  h1 {{ margin-bottom: 4px; }}
  .updated {{ font-family: Arial, sans-serif; color: #4A4A4A; font-size: 0.85rem; margin-bottom: 32px; }}
  table {{ width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; font-size: 0.9rem; margin-bottom: 40px; }}
  th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #D8D3C7; vertical-align: top; }}
  th {{ background: #EFECE4; }}
  img {{ max-width: 100%; border: 1px solid #D8D3C7; border-radius: 6px; margin-bottom: 24px; display: block; }}

  details {{
    border: 1px solid #D8D3C7;
    border-radius: 6px;
    margin-bottom: 12px;
    background: #fff;
  }}
  details summary {{
    font-family: Arial, sans-serif;
    font-weight: bold;
    font-size: 0.92rem;
    padding: 12px 16px;
    cursor: pointer;
    list-style: none;
  }}
  details summary::-webkit-details-marker {{ display: none; }}
  details summary::before {{
    content: "\\25B8";
    display: inline-block;
    margin-right: 8px;
    transition: transform 0.15s ease;
  }}
  details[open] > summary::before {{ transform: rotate(90deg); }}
  details > *:not(summary) {{ padding: 0 16px 16px; }}
  details table {{ margin: 0; }}
  .section {{ margin-bottom: 24px; }}
  .section > summary {{ font-size: 1.25rem; padding: 18px 16px; }}
  summary:focus-visible {{ outline: 3px solid #2C5F4F; outline-offset: 2px; }}
  .toggle-label {{ float: right; font-size: 0.85rem; font-weight: normal; }}
  .toggle-label::after {{ content: "Expand"; }}
  details[open] > summary .toggle-label::after {{ content: "Minimize"; }}
  .table-scroll {{ overflow-x: auto; }}
  .signup-actions {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .signup-actions form {{ flex: 1 1 320px; min-width: 0; }}
  .signup-actions input[type="email"] {{ min-width: 0; }}
  .signup-actions > a {{ text-align: center; }}
  @media (max-width: 540px) {{
    .signup-actions form, .signup-actions > a {{ flex-basis: 100%; }}
  }}
  details ul {{ margin: 0; padding-left: 20px; }}
  .about-me > summary {{ font-size: 1rem; padding: 12px 16px; }}
  .about-me .bio {{ font-size: 1rem; line-height: 1.65; }}
  .about-me .bio p:last-child {{ margin-bottom: 0; }}
</style>
</head>
<body>
  <img src="{photo_src}" alt="Cherry blossoms lining the road at East Potomac Park" style="width:100%; max-height:320px; object-fit:cover; border-radius:8px; margin-bottom:6px;">
  <p style="font-family: Arial, sans-serif; font-size: 0.72rem; color: #4A4A4A; margin: 0 0 28px;">Photo: NPS / NCR Photo Library (public domain)</p>

  <h1>East Potomac Park &mdash; Case Tracker</h1>
  <div class="visit-counter" style="display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin:12px 0; font-family:Arial,sans-serif; font-size:0.85rem; color:#4A4A4A;">
    <a href="https://hits.sh/tbilkitty.github.io/East-Potomac-Park-Tracker/" target="_blank" rel="noopener" aria-label="View page visit statistics" title="Visits, not unique people. Some of these are my dumb ass checking whether I finally fixed the page. Sry.">
      <img src="https://hits.sh/tbilkitty.github.io/East-Potomac-Park-Tracker.svg?style=flat-square&amp;label=Page%20visits&amp;color=2c5f4f&amp;labelColor=333333"
           alt="Page visit counter unavailable" height="24" referrerpolicy="no-referrer"
           style="display:block; height:24px; width:auto; max-width:100%; margin:0; border:0; border-radius:3px;">
    </a>
    <span>Since this counter was added &middot; visits, not unique people</span>
  </div>
  <p class="updated">Automatically updated daily. Last updated: {updated}</p>

  <div style="display:flex; gap:10px; flex-wrap:wrap; font-family: Arial, sans-serif; margin-bottom: 24px;">
    <a href="write-to-congress.html" style="display:inline-block; background:#2C5F4F; color:#fff; text-decoration:none; padding:10px 18px; border-radius:4px; font-weight:bold;">
      Write to Your Member of Congress &rarr;
    </a>

  </div>

  <details class="section" open>
    <summary>News Articles <span class="toggle-label" aria-hidden="true"></span></summary>
    <div>
  <p style="font-family: Arial, sans-serif; font-size: 0.85rem; color: #4A4A4A;">
    All saved articles are retained and ranked by estimated text novelty compared with
    earlier saved articles. This automated score is not a fact-check or a measure of importance.
    The top three appear below; expand the list to see the rest.
  </p>
  <p>{len(ranked_articles)} ranked articles</p>
  <div>{article_rows}</div>

    </div>
  </details>

  <details class="section" open>
    <summary>Court Documents &amp; Public Notices <span class="toggle-label" aria-hidden="true"></span></summary>
    <div>
  <div style="font-family: Arial, sans-serif; font-size: 0.92rem; margin-bottom: 12px;">
    {hearings_section}
  </div>
  <details style="margin-bottom: 32px;">
    <summary>Full Docket &mdash; DC Preservation League v. Department of Interior (1:26-cv-00477) ({len(df_docket)} entries)</summary>
    <div class="table-scroll"><table>
      <tr><th>Date Filed</th><th>Entry #</th><th>Description</th></tr>
      {docket_rows}
    </table></div>
  </details>

    </div>
  </details>

    <details class="section" open>
    <summary>Public Search Interest <span class="toggle-label" aria-hidden="true"></span></summary>
    <div>

      <p style="font-family:Arial,sans-serif;">
        &ldquo;East Potomac Park&rdquo; &middot; United States &middot;
        Past 3 months &middot; Google Trends
      </p>

      {trends_html}

      <p style="font-family:Arial,sans-serif; font-size:0.9rem; color:#4A4A4A;">
        Relative search interest, scaled from 0 to 100:
        100 marks peak interest in this period and region.
        These are not search counts. Low search volume can appear
        as zero or insufficient data. Search activity measures
        attention, not support for a particular outcome.
      </p>

      <p style="font-family:Arial,sans-serif; font-size:0.9rem;">
        Data source:
        <a href="https://trends.google.com/trends/explore?date=today%203-m&amp;geo=US&amp;q=East%20Potomac%20Park&amp;hl=en"
           target="_blank"
           rel="noopener">
          Google Trends &mdash; open the interactive chart
        </a>.
      </p>

    </div>
  </details>

  <details class="section" open>
    <summary>Email Subscription <span class="toggle-label" aria-hidden="true"></span></summary>
    <div style="font-family: Arial, sans-serif;">
    <p style="font-size:0.88rem; color:#4A4A4A; margin:6px 0 12px;">Only sent when there's genuinely new hearing or notice activity &mdash; not a daily digest.</p>
    <div class="signup-actions">
    <form action="https://buttondown.com/api/emails/embed-subscribe/tbilkitty" method="post" target="popupwindow"
          onsubmit="window.open('https://buttondown.com/tbilkitty', 'popupwindow')" style="display:flex; gap:8px; flex-wrap:wrap;">
      <input type="email" name="email" placeholder="you@example.com" required style="flex:1; padding:8px 10px; border:1px solid #D8D3C7; border-radius:4px;">
      <input type="submit" value="Subscribe" style="background:#2C5F4F; color:#fff; border:none; padding:8px 16px; border-radius:4px; font-weight:bold; cursor:pointer;">
    </form>
    <a href="https://github.com/sponsors/tbilkitty" target="_blank" rel="noopener" style="display:inline-block; background:#fff; color:#2C5F4F; border:2px solid #2C5F4F; text-decoration:none; padding:8px 18px; border-radius:4px; font-weight:bold;">
      &#9749; Caffeinate a Broke-Ass Law Student
    </a>
    </div>
    </div>
  </details>
  <details class="section about-me" id="about-me">
    <summary>About Me <span class="toggle-label" aria-hidden="true"></span></summary>
    <div class="bio">
      <p>I&rsquo;m a single parent to a wonderful child. I&rsquo;m also a full time law
      student who works three jobs.</p>
      <p>My legal interest is in using tax policy to advance social equity.
      Go UBalt Law!</p>
      <p>This website is dedicated to those who cherish nature and social justice. <3 </p>
    </div>
  </details>
</body>
</html>"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Page generated successfully.", flush=True)


if __name__ == "__main__":
    main()
