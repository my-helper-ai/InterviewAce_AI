import requests
import pandas as pd
import os
import time


API_URL = "https://api.stackexchange.com/2.3/questions"


def collect_stackoverflow_questions(
    tag="python",
    pages=5,
    pagesize=100
):
    """
    Collect technical questions from Stack Overflow
    using the Stack Exchange API with pagination.
    """

    records = []

    for page in range(1, pages + 1):

        print(f"  Fetching page {page}/{pages}...")

        params = {
            "site": "stackoverflow",
            "tagged": tag,
            "sort": "votes",
            "order": "desc",
            "pagesize": pagesize,
            "page": page
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        for item in data.get("items", []):

            records.append({
                "source": "Stack Overflow",
                "domain": tag,
                "question_id": item.get("question_id"),
                "title": item.get("title"),
                "score": item.get("score"),
                "answer_count": item.get("answer_count"),
                "tags": ", ".join(item.get("tags", [])),
                "link": item.get("link")
            })

        # Stop if API says there are no more pages
        if not data.get("has_more", False):
            break

        # Small delay between API requests
        time.sleep(0.5)

    return pd.DataFrame(records)


if __name__ == "__main__":

    domains = [
        "python",
        "sql",
        "pandas",
        "machine-learning"
    ]

    os.makedirs(
        "data/api_data",
        exist_ok=True
    )

    for domain in domains:

        print(
            f"\nCollecting Stack Overflow data for: {domain}"
        )

        try:

            df = collect_stackoverflow_questions(
                tag=domain,
                pages=5,
                pagesize=100
            )

            output_path = (
                f"data/api_data/"
                f"stackexchange_{domain}.csv"
            )

            df.to_csv(
                output_path,
                index=False,
                encoding="utf-8"
            )

            print(
                f"Collected {len(df)} records."
            )

            print(
                f"Saved to: {output_path}"
            )

        except Exception as e:

            print(
                f"Error collecting {domain}: {e}"
            )