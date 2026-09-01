import os
import json
import time
from datetime import datetime, timezone

import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOCKET_ID = 72278895  # DC Preservation League v. Department of Interior, 1:26-cv-00477
API_TOKEN = os.environ.get("COURTLISTENER_TOKEN", "")
headers = {"Authorization": f"Token {API_TOKEN}"} if API_TOKEN else {}


def get_docket_entries(docket_id):
    entries = []
    url = "https://www.courtlistener.com/api/rest/v4/docket-entries/"
    params = {"docket": docket_id, "order_by": "-date_filed"}
    while url:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            time.sleep(15)
            continue
        resp.raise_for_status()
        data = resp.json()
        entries.extend(data["results"])
        url = data.get("next")
        params = {}
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
        print(f"Federal Register fetch failed: {e}")
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


import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup

ARTICLE_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EastPotomacParkTracker/1.0; +https://github.com)"
}


def fetch_article_text(url, timeout=15, max_chars=4000):
    try:
        resp = requests.get(url, headers=ARTICLE_FETCH_HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
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


def rank_articles_by_novelty(df_media, max_articles=30):
    if df_media.empty:
        return []

    df = df_media.sort_values("seendate", ascending=True).head(max_articles).copy()

    texts = []
    for _, row in df.iterrows():
        text = fetch_article_text(row["url"])
        texts.append(text if text else row.get("title", ""))
        time.sleep(1)  # be polite to the sites we're fetching from

    df["fetched_text"] = texts

    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    tfidf_matrix = vectorizer.fit_transform(df["fetched_text"])

    results = []
    seen_vector = None
    for i in range(len(df)):
        current = tfidf_matrix[i]
        if seen_vector is None:
            novelty = 1.0
        else:
            sim = cosine_similarity(current, seen_vector)[0][0]
            novelty = max(0.0, 1.0 - sim)
        seen_vector = current if seen_vector is None else seen_vector + current

        row = df.iloc[i]
        results.append({
            "title": row.get("title", "(untitled)"),
            "url": row["url"],
            "domain": row.get("domain", ""),
            "seendate": row["seendate"],
            "novelty": novelty,
            "summary": summarize_text(row["fetched_text"]),
        })

    results.sort(key=lambda r: r["novelty"], reverse=True)
    return results


def main():
    os.makedirs("data", exist_ok=True)
    now = datetime.now(timezone.utc)

    # --- Docket ---
    df_docket = pd.DataFrame(columns=["date_filed", "entry_number", "description"])
    try:
        entries = get_docket_entries(DOCKET_ID)
        if entries:
            df_docket = pd.DataFrame(entries)
            df_docket = df_docket[["date_filed", "entry_number", "description"]].sort_values(
                "date_filed", ascending=False
            )
    except requests.exceptions.RequestException as e:
        print(f"CourtListener fetch failed, keeping previous data if any: {e}")
        if os.path.exists("data/east_potomac_docket.csv"):
            df_docket = pd.read_csv("data/east_potomac_docket.csv")
    df_docket.to_csv("data/east_potomac_docket.csv", index=False)

    # --- Upcoming hearings / notices ---
    fr_notices = get_federal_register_notices("East Potomac", "national-park-service")
    fr_notices += get_federal_register_notices("East Potomac", "interior-department")
    hearing_entries = flag_hearing_entries(df_docket)

    # --- Media (rolling 90-day window, GDELT's actual coverage range) ---
    start = (now - pd.Timedelta(days=90)).strftime("%Y%m%d%H%M%S")
    end = now.strftime("%Y%m%d%H%M%S")
    df_media = pd.DataFrame()
    try:
        articles = gdelt_search('"East Potomac" (cherry OR golf)', start, end)
        if articles:
            df_media = pd.DataFrame(articles)
            df_media["seendate"] = pd.to_datetime(df_media["seendate"])
    except Exception as e:
        print(f"GDELT fetch failed, keeping previous data if any: {e}")
        if os.path.exists("data/east_potomac_media.csv"):
            df_media = pd.read_csv("data/east_potomac_media.csv")
            if not df_media.empty:
                df_media["seendate"] = pd.to_datetime(df_media["seendate"])
    df_media.to_csv("data/east_potomac_media.csv", index=False)

    # --- Charts ---
    if not df_media.empty:
        weekly = df_media.set_index("seendate").resample("W").size()
        plt.figure(figsize=(8, 4))
        weekly.plot(kind="bar", color="#E03C31")
        plt.title("Weekly Coverage Volume \u2014 East Potomac Park")
        plt.ylabel("Articles")
        plt.tight_layout()
        plt.savefig("media_coverage_over_time.png", dpi=150)
        plt.close()

    ranked_articles = rank_articles_by_novelty(df_media, max_articles=30)

    # --- Render index.html ---
    updated = now.strftime("%B %d, %Y at %H:%M UTC")

    docket_rows = ""
    for _, row in df_docket.head(30).iterrows():
        docket_rows += (
            f"<tr><td>{row['date_filed']}</td><td>{row['entry_number']}</td>"
            f"<td>{row['description']}</td></tr>\n"
        )
    if not docket_rows:
        docket_rows = "<tr><td colspan='3'>No docket entries retrieved yet.</td></tr>"

    hearing_rows = ""
    for _, row in hearing_entries.iterrows():
        hearing_rows += f"<li><strong>{row['date_filed']}</strong> \u2014 {row['description']}</li>\n"

    notice_rows = ""
    for doc in fr_notices:
        title = doc.get("title", "")
        pub_date = doc.get("publication_date", "")
        html_url = doc.get("html_url", "#")
        notice_rows += f'<li><strong>{pub_date}</strong> \u2014 <a href="{html_url}" target="_blank" rel="noopener">{title}</a></li>\n'

    if not hearing_rows and not notice_rows:
        hearings_section = "<p>No public hearings or notices found in the current data. This section updates daily.</p>"
    else:
        hearings_section = ""
        if hearing_rows:
            hearings_section += f"<h3>Court Hearings &amp; Conferences</h3><ul>{hearing_rows}</ul>"
        if notice_rows:
            hearings_section += f"<h3>Federal Register Notices</h3><ul>{notice_rows}</ul>"

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

    charts_html = ""
    if not df_media.empty:
        charts_html = '<img src="media_coverage_over_time.png" alt="Weekly coverage volume chart">'
    else:
        charts_html = "<p>Media coverage data not available this run.</p>"

    article_rows = ""
    for a in ranked_articles:
        pct = round(a["novelty"] * 100)
        date_str = a["seendate"].strftime("%b %d, %Y") if hasattr(a["seendate"], "strftime") else str(a["seendate"])
        article_rows += f"""
        <div style="border-bottom:1px solid #D8D3C7; padding:16px 0;">
          <div style="display:flex; justify-content:space-between; align-items:baseline; gap:12px;">
            <a href="{a['url']}" target="_blank" rel="noopener" style="font-weight:bold; color:#1A1A1A; text-decoration:none;">{a['title']}</a>
            <span style="font-family:Arial,sans-serif; font-size:0.78rem; font-weight:bold; color:#2C5F4F; white-space:nowrap;">{pct}% new info</span>
          </div>
          <div style="font-family:Arial,sans-serif; font-size:0.78rem; color:#4A4A4A; margin:2px 0 8px;">{a['domain']} &middot; {date_str}</div>
          <div style="font-size:0.92rem; color:#333;">{a['summary']}</div>
        </div>
        """
    if not article_rows:
        article_rows = "<p>No articles available to rank this run.</p>"

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
</style>
</head>
<body>
  <h1>East Potomac Park &mdash; Case Tracker</h1>
  <p class="updated">Automatically updated daily. Last updated: {updated}</p>

  <p style="font-family: Arial, sans-serif;">
    <a href="write-to-congress.html" style="display:inline-block; background:#2C5F4F; color:#fff; text-decoration:none; padding:10px 18px; border-radius:4px; font-weight:bold;">
      Write to Your Member of Congress &rarr;
    </a>
  </p>

  <h2>Upcoming Hearings &amp; Public Notices</h2>
  <div style="font-family: Arial, sans-serif; font-size: 0.92rem; margin-bottom: 32px;">
    {hearings_section}
  </div>

  <div style="font-family: Arial, sans-serif; background:#EFECE4; border:1px solid #D8D3C7; border-radius:6px; padding:20px; margin-bottom:32px;">
    <strong>Get email updates</strong>
    <p style="font-size:0.88rem; color:#4A4A4A; margin:6px 0 12px;">Only sent when there's genuinely new hearing or notice activity &mdash; not a daily digest.</p>
    <form action="https://buttondown.com/api/emails/embed-subscribe/YOUR-BUTTONDOWN-USERNAME" method="post" target="popupwindow"
          onsubmit="window.open('https://buttondown.com/YOUR-BUTTONDOWN-USERNAME', 'popupwindow')" style="display:flex; gap:8px;">
      <input type="email" name="email" placeholder="you@example.com" required style="flex:1; padding:8px 10px; border:1px solid #D8D3C7; border-radius:4px;">
      <input type="submit" value="Subscribe" style="background:#2C5F4F; color:#fff; border:none; padding:8px 16px; border-radius:4px; font-weight:bold; cursor:pointer;">
    </form>
  </div>

  <h2>Media Coverage</h2>
  {charts_html}

  <p style="font-family: Arial, sans-serif; font-size: 0.85rem; color: #4A4A4A; margin-top: 24px;">
    Articles below are ranked by how much genuinely new information each one adds, compared to everything
    already covered by earlier articles &mdash; not just by outlet or recency.
  </p>
  <div style="margin-bottom: 40px;">
    {article_rows}
  </div>

  <h2>Docket &mdash; DC Preservation League v. Department of Interior (1:26-cv-00477)</h2>
  <table>
    <tr><th>Date Filed</th><th>Entry #</th><th>Description</th></tr>
    {docket_rows}
  </table>
</body>
</html>"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
