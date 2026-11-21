import requests
import pandas as pd
import os


TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"


def collect_hackernews_stories(limit=20):

    response = requests.get(
        TOP_STORIES_URL,
        timeout=20
    )

    response.raise_for_status()

    story_ids = response.json()[:limit]

    records = []

    for story_id in story_ids:

        item_response = requests.get(
            ITEM_URL.format(story_id),
            timeout=20
        )

        if item_response.status_code != 200:
            continue

        item = item_response.json()

        if item.get("type") != "story":
            continue

        records.append({
            "source": "Hacker News",
            "story_id": item.get("id"),
            "title": item.get("title"),
            "score": item.get("score"),
            "author": item.get("by"),
            "comments": item.get("descendants", 0),
            "url": item.get("url")
        })

    return pd.DataFrame(records)


if __name__ == "__main__":

    print("\nCollecting technology news from Hacker News...")

    df = collect_hackernews_stories(
        limit=30
    )

    os.makedirs(
        "data/api_data",
        exist_ok=True
    )

    output_path = (
        "data/api_data/hackernews_technology.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Collected {len(df)} stories."
    )

    print(
        f"Saved to: {output_path}"
    )

    print("\nSample data:")
    print(df.head())