# SpotifyScraper

Utilities for discovering large batches of Spotify artist profile URLs without the official API and then analyzing the exported artist metadata. The project is organized around two standalone scripts:

- `discover_artists.py` — launches Playwright-driven Chromium workers that crawl Spotify search results, persist the unique artist profile URLs, and resume from prior runs.
- `analyze_artists.py` — inspects one or more JSON exports, extracts contact info, filters by geography and audience size, and produces a ranked CSV report.

## Project Structure

| Path | Purpose |
| --- | --- |
| `discover_artists.py` | Main scraper/orchestrator with extensive CLI flags for query planning and concurrency control. |
| `urls.txt` | Rolling list of artist profile URLs gathered by the scraper (used for resume/fresh runs). |
| `JSON Files/` | Optional holding area for raw JSON exports pulled from the scraper or other sources. The analyzer can ingest the entire directory. |
| `analyze_artists.py` | Analyzer/CLI utility that generates contact lists and the derived `artist_popularity.csv`. |
| `artist_popularity.csv` | Latest popularity ranking exported by the analyzer (regenerated each run). |
| `.gitignore` | Prevents committing local environments, caches, and Playwright output. |
| `requirements.txt` | Python dependency list (currently just Playwright). |

> **Note:** The bulk JSON exports in `JSON Files/` are produced via the companion [Apify Spotify Artist scraper](https://console.apify.com/organization/aCJfkP6SwfOYzFvJv/actors/bLKdJEP8h5UUWK5Jy/input). Generate or refresh datasets there, download the JSON output, and drop it into this folder for analysis.

## Prerequisites

- Python 3.10+ recommended.
- Node.js **not** required; Playwright’s Python package bundles Chromium.
- macOS users should ensure Xcode command-line tools are installed for Playwright.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
playwright install  # downloads the Chromium browser used by the scraper
```

## Discovering Artist URLs

### Basic usage

```bash
python discover_artists.py --count 1000 --fresh
```

- `--count` – number of unique artist URLs to collect (default 1000).
- `--fresh` – ignore the existing `urls.txt` so the run does not resume.
- `--out` – change the destination file.

Key performance flags have sensible defaults tuned for an Apple M3 Pro (see `DiscoverConfig`), but you can override them:

| Flag | Description |
| --- | --- |
| `--workers` | Number of Playwright contexts to spawn (default 6). |
| `--per-query-target` | Stop scrolling a search term once this many new artists are found. |
| `--max-scrolls`, `--scroll-pause-ms`, `--initial-wait-ms` | Control page traversal depth and pacing. |
| `--deterministic-length`, `--random-min/max/cap` | Shape the query planner’s search-space exploration. |
| `--fresh` vs default resume | Decide whether to append to or overwrite `urls.txt`. |

Logs show per-worker progress, warnings, and stall handling. Network hiccups (e.g., `net::ERR_NETWORK_CHANGED`) are automatically retried; no manual action is needed unless they become persistent.

## Analyzing Artist Metadata

`analyze_artists.py` reads either a single JSON file or every `.json` file inside a directory and produces contact-focused summaries plus a popularity CSV.

### Example commands

```bash
# Analyze a single export
python analyze_artists.py spotify_artist_info_1-500.json

# Aggregate every JSON file in the "JSON Files" directory
python analyze_artists.py "JSON Files"
```

### What it outputs

- Lists every artist with available Instagram or Gmail contact.
- Filters those artists to ones appearing in US top cities.
- Applies min/max audience thresholds (defaults: >500 followers & >5k monthly listeners, up to 100k followers & 1M listeners) and prints the qualifying subset.
- Generates `artist_popularity.csv` with normalized popularity scores for the filtered artists.

Customize the thresholds by editing `instagram_artists_in_country_with_audience` parameters in the script or wrapping the function in your own CLI.

## Data Handling Tips

- Keep `urls.txt` under version control to track scraping progress, or add it to `.gitignore` if it grows very large.
- Large JSON exports can live in `JSON Files/` to keep the root tidy. The analyzer automatically concatenates all `.json` files in that directory.
- Consider rotating JSON exports into dated sub-folders once they get too large for easy diffing.

## Getting Ready to Push

- Run `python -m compileall .` or simply execute both scripts once to ensure no runtime errors.
- Confirm `.venv/`, Playwright artifacts, and OS cruft stay ignored via `.gitignore`.
- Update `README.md` if you introduce new scripts or flags so collaborators know how to run them.

With the README, requirements, and ignore rules in place, the repository is ready for `git init`, `git add .`, and the first GitHub push. Adjust the documentation as your scraping or analysis workflows evolve.
