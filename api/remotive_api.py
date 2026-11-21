import requests
import pandas as pd
import os


API_URL = "https://remotive.com/api/remote-jobs"


def collect_jobs(category="software-dev", limit=100):

    params = {
        "category": category
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    jobs = data.get("jobs", [])[:limit]

    records = []

    for job in jobs:

        records.append({
            "source": "Remotive",
            "job_id": job.get("id"),
            "title": job.get("title"),
            "company": job.get("company_name"),
            "category": job.get("category"),
            "job_type": job.get("job_type"),
            "location": job.get("candidate_required_location"),
            "publication_date": job.get("publication_date"),
            "description": job.get("description"),
            "url": job.get("url")
        })

    return pd.DataFrame(records)


if __name__ == "__main__":

    categories = [
        "software-dev",
        "data",
        "devops",
        "cloud",
        "customer-support"
    ]

    os.makedirs(
        "data/api_data",
        exist_ok=True
    )

    for category in categories:

        print(
            f"\nCollecting jobs for category: {category}"
        )

        try:

            df = collect_jobs(
                category=category,
                limit=100
            )

            safe_category = category.replace(
                "-", "_"
            )

            output_path = (
                f"data/api_data/"
                f"remotive_{safe_category}.csv"
            )

            df.to_csv(
                output_path,
                index=False,
                encoding="utf-8"
            )

            print(
                f"Collected {len(df)} jobs."
            )

            print(
                f"Saved to: {output_path}"
            )

        except Exception as e:

            print(
                f"Could not collect {category}: {e}"
            )