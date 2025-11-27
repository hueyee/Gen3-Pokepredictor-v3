"""Data scraper for Pokemon Showdown replays."""
import os
from pathlib import Path
import requests
import re
import time
from datetime import timedelta

import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def fetch_replays(format_name="gen3ou", pages=1):
    """Fetch replays from the Pokemon Showdown API."""
    base_url = f"https://replay.pokemonshowdown.com/search.json?format={format_name}&page="
    all_replays = []

    for page in range(1, pages + 1):
        print(f"\rFetching page {page}/{pages}...", end="", flush=True)
        response = requests.get(base_url + str(page))
        if response.status_code == 200:
            replays = response.json()
            all_replays.extend(replays)
        else:
            print(f"\nFailed to fetch page {page}: {response.status_code}")

    print()  # Move to the next line after progress
    return all_replays


def sanitize_filename(filename):
    """Remove or replace invalid characters in a filename."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


def download_replay(replay_id, player1, player2, stats, output_dir):
    """Download the replay log and save as an HTML file."""
    log_url = f"https://replay.pokemonshowdown.com/{replay_id}.log"
    response = requests.get(log_url)

    if response.status_code == 200:
        log_content = (
            f"<script type=\"text/plain\" class=\"battle-log-data\">\n"
            f"{response.text}\n"
            f"</script>"
        )
        sanitized_player1 = sanitize_filename(player1)
        sanitized_player2 = sanitize_filename(player2)
        filename = output_dir / f"{replay_id}-{sanitized_player1}-{sanitized_player2}.html"

        with open(filename, "w", encoding="utf-8") as file:
            file.write(log_content)

        file_size = os.path.getsize(filename) / 1024  # Convert to KB
        stats['total_size'] += file_size
        stats['saved'] += 1
        return file_size
    else:
        stats['failed'] += 1
        return None


def display_progress_bar(percentage, bar_length=30):
    """Display a progress bar in the console."""
    progress = int(bar_length * (percentage / 100))
    bar = "=" * progress + "-" * (bar_length - progress)
    return f"[{bar}] {percentage:.2f}%"


def main():
    config = load_config()
    raw_path = Path(config['data']['raw_path'])
    replays_dir = raw_path / "Replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    format_name = "gen3ou"
    pages_to_fetch = 160
    start_time = time.time()

    replays = fetch_replays(format_name=format_name, pages=pages_to_fetch)
    total_replays = len(replays)
    stats = {"processed": 0, "saved": 0, "skipped": 0, "failed": 0, "total_size": 0}

    for replay in replays:
        replay_id = replay.get("id")
        players = replay.get("players", [])
        rating = replay.get("rating")

        stats["processed"] += 1
        if replay_id and len(players) == 2 and rating is not None:
            player1, player2 = players
            file_size = download_replay(replay_id, player1, player2, stats, replays_dir)
        else:
            stats["skipped"] += 1
            file_size = None

        elapsed_time = time.time() - start_time
        percentage_complete = (stats["processed"] / total_replays) * 100
        progress_bar = display_progress_bar(percentage_complete)

        print(
            f"\r{progress_bar} | Processed: {stats['processed']}/{total_replays} | "
            f"Saved: {stats['saved']} | Skipped: {stats['skipped']} "
            f"({(stats['skipped'] / stats['processed']) * 100:.2f}%) | "
            f"Failed: {stats['failed']} | Last File Size: {file_size or 0:.2f} KB | "
            f"Total Size: {stats['total_size']:.2f} KB | "
            f"Time Elapsed: {timedelta(seconds=int(elapsed_time))}",
            end="", flush=True
        )

    print()
    total_time = timedelta(seconds=int(time.time() - start_time))
    print(f"Task completed in {total_time}. Total saved: {stats['saved']}. Total size: {stats['total_size']:.2f} KB.")


if __name__ == "__main__":
    main()
