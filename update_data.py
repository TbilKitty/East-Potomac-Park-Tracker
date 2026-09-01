import os
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

        top_domains = df_media["domain"].value_counts().head(10)
        plt.figure(figsize=(8, 4))
        top_domains.plot(kind="barh", color="#20917a")
        plt.title("Top Outlets Covering the Story")
        plt.tight_layout()
        plt.savefig("top_outlets.png", dpi=150)
        plt.close()

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

    charts_html = ""
    if not df_media.empty:
        charts_html = (
            '<img src="media_coverage_over_time.png" alt="Weekly coverage volume chart">\n'
            '<img src="top_outlets.png" alt="Top outlets chart">'
        )
    else:
        charts_html = "<p>Media coverage data not available this run.</p>"

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

  <h2>Media Coverage</h2>
  {charts_html}

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
