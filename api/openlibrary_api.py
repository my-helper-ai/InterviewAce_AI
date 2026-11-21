import requests
import pandas as pd
import os


API_URL = "https://openlibrary.org/search.json"


def collect_openlibrary_books(topics, limit=10):

    records = []

    for topic in topics:

        params = {
            "q": topic,
            "limit": limit
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        for book in data.get("docs", []):

            authors = book.get("author_name", [])

            records.append({
                "source": "Open Library",
                "topic": topic,
                "title": book.get("title"),
                "author": ", ".join(authors[:3]),
                "first_publish_year": book.get(
                    "first_publish_year"
                ),
                "subjects": ", ".join(
                    book.get("subject", [])[:5]
                ),
                "isbn": (
                    book.get("isbn", [None])[0]
                    if book.get("isbn")
                    else None
                )
            })

    return pd.DataFrame(records)


if __name__ == "__main__":

    topics = [
        "Python programming",
        "SQL",
        "machine learning",
        "data analysis",
        "cloud computing"
    ]

    print("\nCollecting learning resources from Open Library...")

    df = collect_openlibrary_books(
        topics,
        limit=10
    )

    os.makedirs(
        "data/api_data",
        exist_ok=True
    )

    output_path = (
        "data/api_data/openlibrary_books.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Collected {len(df)} book records."
    )

    print(
        f"Saved to: {output_path}"
    )

    print("\nSample data:")
    print(
        df[
            ["topic", "title", "author"]
        ].head()
    )