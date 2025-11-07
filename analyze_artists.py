#!/usr/bin/env python3
"""
Simple utilities for inspecting the Spotify artist JSON export.
"""
# Run `pip install -r requirements.txt` before executing this script.

from __future__ import annotations

import csv
import html
import json
import re
import sys
from pathlib import Path

EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@gmail\.com\b", re.IGNORECASE)
MAX_FOLLOWERS = 100_000
MAX_MONTHLY_LISTENERS = 1_000_000


def parse_metric(value) -> int:
    """
    Convert follower/listener values to integers, falling back to zero on errors.
    """
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_artist_data(path: Path) -> list[dict]:
    """
    Load the JSON file and ensure we end up with a list of artist records.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit(f"Expected a list at the top level of {path}, got {type(data).__name__}")

    return data


def extract_contact_info(entry: dict) -> dict:
    """
    Determine available contact information (Instagram URL, Gmail addresses).
    """
    instagram_url = entry.get("instagram_url") or ""
    biography = entry.get("biography") or ""
    decoded_bio = html.unescape(biography)
    emails = EMAIL_PATTERN.findall(decoded_bio)
    normalized_emails = sorted({email.lower() for email in emails})
    return {
        "instagram_url": instagram_url,
        "emails": normalized_emails,
    }


def has_contact(contact_info: dict) -> bool:
    """
    Determine if the parsed contact info contains at least one channel.
    """
    return bool(contact_info.get("instagram_url") or contact_info.get("emails"))


def artist_in_country(entry: dict, country_code: str) -> bool:
    """
    Check whether any of the artist's top cities are in the requested country.
    """
    top_cities = entry.get("top_cities")
    if not isinstance(top_cities, list):
        return False

    for city in top_cities:
        if isinstance(city, dict) and city.get("country") == country_code:
            return True
    return False


def build_contact_entry(entry: dict, include_metrics: bool = False) -> dict | None:
    """
    Build a reusable contact payload; optionally include follower metrics.
    """
    contact = extract_contact_info(entry)
    if not has_contact(contact):
        return None

    payload: dict = {
        "artist_name": entry.get("artist_name") or "<unknown>",
        "instagram_url": contact["instagram_url"],
        "emails": contact["emails"],
    }

    if include_metrics:
        payload["followers"] = entry.get("followers") or 0
        payload["monthly_listeners"] = entry.get("monthly_listeners") or 0

    return payload


def format_contact_details(entry: dict) -> str:
    """
    Produce a human-readable representation of the available contact channels.
    """
    contacts = [
        value
        for value in (
            entry.get("instagram_url"),
            "; ".join(entry.get("emails", [])) if entry.get("emails") else "",
        )
        if value
    ]
    return " | ".join(contacts) if contacts else "<no contact>"


def artists_with_contact_info(data: list[dict]) -> list[dict]:
    """
    Extract contact info for entries that include an Instagram link or Gmail address.
    """
    contact_entries: list[dict] = []
    for entry in data:
        payload = build_contact_entry(entry)
        if payload:
            contact_entries.append(payload)

    return contact_entries


def instagram_artists_in_country(data: list[dict], country_code: str) -> list[dict]:
    """
    Limit contact-enabled artists to those whose top cities include the target country.
    """
    matches: list[dict] = []
    for entry in data:
        if not artist_in_country(entry, country_code):
            continue

        payload = build_contact_entry(entry)
        if payload:
            matches.append(payload)

    return matches


def _is_within_audience_bounds(
    follower_value: int,
    monthly_value: int,
    min_followers: int,
    min_monthly_listeners: int,
    max_followers: int | None,
    max_monthly_listeners: int | None,
) -> bool:
    if follower_value <= min_followers or monthly_value <= min_monthly_listeners:
        return False
    if max_followers is not None and follower_value > max_followers:
        return False
    if max_monthly_listeners is not None and monthly_value > max_monthly_listeners:
        return False
    return True


def instagram_artists_in_country_with_audience(
    data: list[dict],
    country_code: str,
    min_followers: int,
    min_monthly_listeners: int,
    max_followers: int | None = None,
    max_monthly_listeners: int | None = None,
) -> list[dict]:
    """
    Further restrict contact-enabled artists in a country to those clearing follower and listener thresholds.
    """
    qualified: list[dict] = []
    for entry in data:
        if not artist_in_country(entry, country_code):
            continue

        payload = build_contact_entry(entry, include_metrics=True)
        if not payload:
            continue

        follower_value = parse_metric(payload["followers"])
        monthly_value = parse_metric(payload["monthly_listeners"])

        if _is_within_audience_bounds(
            follower_value,
            monthly_value,
            min_followers,
            min_monthly_listeners,
            max_followers,
            max_monthly_listeners,
        ):
            qualified.append(payload)

    return qualified


def compute_popularity_rankings(
    data: list[dict],
    filtered_entries: list[dict] | None = None,
) -> list[dict]:
    """
    Compute popularity scores for a set of artists based on follower and listener normalization.
    """
    if filtered_entries is None:
        dataset: list[dict] = []
        for entry in data:
            payload = build_contact_entry(entry, include_metrics=True)
            if payload:
                dataset.append(payload)
    else:
        dataset = [dict(item) for item in filtered_entries]

    follower_values = [row["followers"] for row in dataset if row["followers"] and row["followers"] > 0]
    monthly_values = [row["monthly_listeners"] for row in dataset if row["monthly_listeners"] and row["monthly_listeners"] > 0]

    min_followers = min(follower_values, default=1)
    min_monthly = min(monthly_values, default=1)

    rankings: list[dict] = []
    for row in dataset:
        follower_component = row["followers"] / min_followers if min_followers else 0
        monthly_component = row["monthly_listeners"] / min_monthly if min_monthly else 0
        popularity_score = (follower_component + monthly_component) / 2

        rankings.append(
            {
                "artist_name": row["artist_name"],
                "followers": row["followers"],
                "monthly_listeners": row["monthly_listeners"],
                "instagram_url": row["instagram_url"],
                "emails": row["emails"],
                "popularity_score": popularity_score,
            }
        )

    rankings.sort(key=lambda item: item["popularity_score"], reverse=True)
    return rankings


def write_popularity_csv(rankings: list[dict], output_path: Path) -> None:
    """
    Write the popularity rankings to a CSV file ordered by score.
    """
    fieldnames = [
        "artist_name",
        "followers",
        "monthly_listeners",
        "popularity_score",
        "instagram_url",
        "email",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rankings:
            csv_row = dict(row)
            emails = csv_row.pop("emails", [])
            csv_row["email"] = "; ".join(emails) if emails else ""
            writer.writerow(csv_row)


def load_from_path(target: Path) -> tuple[list[dict], list[Path]]:
    """
    Load artist data from a single JSON file or every JSON file inside a directory.
    """
    if target.is_dir():
        files = sorted(path for path in target.glob("*.json") if path.is_file())
        if not files:
            raise SystemExit(f"No JSON files found inside directory: {target}")
        aggregated: list[dict] = []
        for file_path in files:
            file_data = load_artist_data(file_path)
            aggregated.extend(file_data)
        return aggregated, files

    if not target.exists():
        raise SystemExit(f"File not found: {target}")

    return load_artist_data(target), [target]


def main() -> None:
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("JSON Files")

    data, source_files = load_from_path(target_path)
    if len(source_files) == 1:
        print(f"Loaded {len(data)} artist entries from {source_files[0]}")
    else:
        print(f"Loaded {len(data)} artist entries from {len(source_files)} files in {target_path}")

    contact_entries = artists_with_contact_info(data)
    print(f"Artists with Instagram links or Gmail contact ({len(contact_entries)} total):")
    for entry in contact_entries:
        print(f"- {entry['artist_name']}: {format_contact_details(entry)}")

    contact_us_entries = instagram_artists_in_country(data, "US")
    print(
        "\nArtists with Instagram links or Gmail contact appearing in US top cities "
        f"({len(contact_us_entries)} total):"
    )
    for entry in contact_us_entries:
        print(f"- {entry['artist_name']}: {format_contact_details(entry)}")

    us_audience_contacts = instagram_artists_in_country_with_audience(
        data,
        country_code="US",
        min_followers=500,
        min_monthly_listeners=5_000,
        max_followers=MAX_FOLLOWERS,
        max_monthly_listeners=MAX_MONTHLY_LISTENERS,
    )
    print(
        "\nUS artists with Instagram or Gmail contact and >500 followers plus >10,000 monthly listeners "
        f"(and <= {MAX_FOLLOWERS:,} followers / <= {MAX_MONTHLY_LISTENERS:,} monthly listeners) "
        f"({len(us_audience_contacts)} total):"
    )
    for entry in us_audience_contacts:
        print(
            f"- {entry['artist_name']}: {format_contact_details(entry)} "
            f"(followers={entry['followers']}, monthly_listeners={entry['monthly_listeners']})"
        )

    popularity_rankings = compute_popularity_rankings(data, filtered_entries=us_audience_contacts)
    output_csv = target_path.with_name("artist_popularity.csv")
    write_popularity_csv(popularity_rankings, output_csv)
    if popularity_rankings:
        print(
            "\nPopularity rankings for US artists with Instagram or Gmail contact and >500 followers plus "
            f">10,000 monthly listeners saved to {output_csv} "
            f"(top artist: {popularity_rankings[0]['artist_name']})"
        )
    else:
        print(
            "\nPopularity rankings for US artists with Instagram or Gmail contact and >500 followers plus "
            f">10,000 monthly listeners saved to {output_csv} (no artist records found)"
        )


if __name__ == "__main__":
    main()
