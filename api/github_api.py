import requests
import pandas as pd
import os


API_URL = "https://api.github.com/search/repositories"


def collect_github_repositories(
    topic="python",
    per_page=20
):
    """
    Collect public GitHub repositories related
    to a selected technology/topic.
    """

    params = {
        "q": topic,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page
    }

    headers = {
        "Accept": "application/vnd.github+json"
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    records = []

    for item in data.get("items", []):

        records.append({
            "source": "GitHub",
            "topic": topic,
            "repository": item.get("full_name"),
            "description": item.get("description"),
            "language": item.get("language"),
            "stars": item.get("stargazers_count"),
            "forks": item.get("forks_count"),
            "url": item.get("html_url")
        })

    return pd.DataFrame(records)


if __name__ == "__main__":

    topics = [
        "python",
        "sql",
        "machine-learning",
        "data-analysis"
    ]

    os.makedirs(
        "data/api_data",
        exist_ok=True
    )

    for topic in topics:

        print(f"\nCollecting GitHub data for: {topic}")

        df = collect_github_repositories(
            topic=topic,
            per_page=20
        )

        safe_topic = topic.replace("-", "_")

        output_path = (
            f"data/api_data/github_{safe_topic}.csv"
        )

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8"
        )

        print(
            f"Collected {len(df)} repositories."
        )

        print(
            f"Saved to: {output_path}"
        )