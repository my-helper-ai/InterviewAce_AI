from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="InterviewAce AI | Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
API_DIR = DATA_DIR / "api_data"
ML_DIR = DATA_DIR / "ml_results"

st.title("📊 InterviewAce AI — Analytics Dashboard")
st.caption(
    "Explore API data collection, technology activity, "
    "data fusion, and machine learning results."
)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

@st.cache_data
def load_csv(path):
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def show_missing_file_message(path):
    st.warning(f"Data file not found: {path}")


# --------------------------------------------------
# Load existing datasets
# --------------------------------------------------

github = load_csv(API_DIR / "github_python.csv")
stack = load_csv(API_DIR / "stackexchange_python.csv")
books = load_csv(API_DIR / "openlibrary_books.csv")
hackernews = load_csv(API_DIR / "hackernews_technology.csv")

model_results = load_csv(ML_DIR / "model_evaluation.csv")
classification_results = load_csv(
    ML_DIR / "classification_evaluation.csv"
)

# Load the fused workbook if available
fused_path = DATA_DIR / "InterviewAce_Data_Fusion.xlsx"

if fused_path.exists():
    try:
        fused = pd.read_excel(fused_path, sheet_name="Fused_Data")
    except Exception:
        fused = pd.DataFrame()
else:
    fused = pd.DataFrame()

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

available_sources = [
    name
    for name, df in [
        ("GitHub", github),
        ("Stack Exchange", stack),
        ("Open Library", books),
        ("Hacker News", hackernews),
    ]
    if not df.empty
]

selected_sources = st.sidebar.multiselect(
    "Select API sources",
    options=available_sources,
    default=available_sources,
)

# --------------------------------------------------
# Summary metrics
# --------------------------------------------------

source_data = {
    "GitHub": github,
    "Stack Exchange": stack,
    "Open Library": books,
    "Hacker News": hackernews,
}

source_counts = {
    name: len(df)
    for name, df in source_data.items()
    if name in selected_sources
}

total_selected_records = sum(source_counts.values())

unique_jobs = 0
remotive_path = API_DIR / "remotive_software_dev.csv"

if remotive_path.exists():
    remotive = load_csv(remotive_path)

    if not remotive.empty:
        if "id" in remotive.columns:
            unique_jobs = remotive["id"].nunique()
        elif "url" in remotive.columns:
            unique_jobs = remotive["url"].nunique()
        else:
            unique_jobs = len(remotive.drop_duplicates())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Selected API Records", f"{total_selected_records:,}")
col2.metric("GitHub Repositories", f"{len(github):,}")
col3.metric("Stack Exchange Questions", f"{len(stack):,}")
col4.metric("Unique Jobs in One Job File", f"{unique_jobs:,}")

st.divider()

# --------------------------------------------------
# Chart 1: Records by API source
# --------------------------------------------------

st.subheader("1. Records Collected by API Source")

if source_counts:
    source_df = pd.DataFrame(
        {
            "API Source": list(source_counts.keys()),
            "Records": list(source_counts.values()),
        }
    )

    fig = px.bar(
        source_df,
        x="API Source",
        y="Records",
        color="API Source",
        text="Records",
        title="Records Available in Selected API Files",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one API source from the sidebar.")

# --------------------------------------------------
# Chart 2: GitHub repository stars
# --------------------------------------------------

st.subheader("2. GitHub Repository Popularity")

if not github.empty and {"repository", "stars"}.issubset(github.columns):
    top_github = github.nlargest(10, "stars")

    fig = px.bar(
        top_github.sort_values("stars"),
        x="stars",
        y="repository",
        orientation="h",
        title="Top Python Repositories by Stars",
        labels={"stars": "GitHub Stars", "repository": "Repository"},
        hover_data=["forks"] if "forks" in github.columns else None,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    show_missing_file_message(API_DIR / "github_python.csv")

# --------------------------------------------------
# Chart 3: GitHub stars distribution
# --------------------------------------------------

st.subheader("3. Distribution of GitHub Stars")

if not github.empty and "stars" in github.columns:
    fig = px.box(
        github,
        y="stars",
        points="all",
        title="Spread of Stars Across Python Repositories",
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Chart 4: Stack Exchange scores
# --------------------------------------------------

st.subheader("4. Stack Exchange Question Activity")

if not stack.empty and {"score", "answer_count"}.issubset(stack.columns):
    fig = px.scatter(
        stack,
        x="answer_count",
        y="score",
        hover_data=["title"],
        title="Question Score vs. Number of Answers",
        labels={
            "answer_count": "Number of Answers",
            "score": "Question Score",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Chart 5: Stack Exchange score distribution
# --------------------------------------------------

st.subheader("5. Distribution of Stack Exchange Scores")

if not stack.empty and "score" in stack.columns:
    fig = px.histogram(
        stack,
        x="score",
        nbins=25,
        title="Distribution of Python Question Scores",
        labels={"score": "Question Score"},
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Chart 6: Hacker News activity
# --------------------------------------------------

st.subheader("6. Hacker News Story Engagement")

if not hackernews.empty and {"title", "score", "comments"}.issubset(
    hackernews.columns
):
    top_stories = hackernews.nlargest(10, "score")

    fig = px.bar(
        top_stories.sort_values("score"),
        x="score",
        y="title",
        orientation="h",
        hover_data=["comments"],
        title="Top Technology Stories by Score",
        labels={"score": "Story Score", "title": "Story"},
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Chart 7: Machine learning MAE comparison
# --------------------------------------------------

st.subheader("7. Regression Model Comparison — MAE")

if not model_results.empty and {"model", "MAE"}.issubset(
    model_results.columns
):
    fig = px.bar(
        model_results.sort_values("MAE"),
        x="model",
        y="MAE",
        text_auto=".2f",
        title="Mean Absolute Error by Regression Model",
        labels={"model": "Model", "MAE": "Mean Absolute Error"},
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Chart 8: Machine learning R² comparison
# --------------------------------------------------

st.subheader("8. Regression Model Comparison — R²")

if not model_results.empty and {"model", "R2"}.issubset(
    model_results.columns
):
    fig = px.bar(
        model_results.sort_values("R2"),
        x="model",
        y="R2",
        text_auto=".3f",
        title="R² Score by Regression Model",
        labels={"model": "Model", "R2": "R² Score"},
    )
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Chart 9: Classification performance
# --------------------------------------------------

st.subheader("9. Classification Model Performance")

if not classification_results.empty:
    metric_columns = [
        column
        for column in ["Accuracy", "Precision", "Recall", "F1"]
        if column in classification_results.columns
    ]

    if metric_columns:
        plot_data = classification_results.melt(
            id_vars=["model"] if "model" in classification_results.columns else None,
            value_vars=metric_columns,
            var_name="Metric",
            value_name="Score",
        )

        fig = px.bar(
            plot_data,
            x="Metric",
            y="Score",
            color="Metric",
            text_auto=".3f",
            title="Classification Evaluation Metrics",
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
else:
    show_missing_file_message(ML_DIR / "classification_evaluation.csv")

# --------------------------------------------------
# Optional: Fused dataset preview
# --------------------------------------------------

st.divider()
st.subheader("Fused Dataset Preview")

if not fused.empty:
    st.write(f"Fused workbook records available: {len(fused):,}")
    st.dataframe(fused.head(20), use_container_width=True)

    st.download_button(
        label="Download Fused Dataset Preview",
        data=fused.head(100).to_csv(index=False).encode("utf-8"),
        file_name="InterviewAce_fused_data_preview.csv",
        mime="text/csv",
    )
else:
    st.info(
        "The fused workbook preview could not be loaded. "
        "The API and ML charts can still be used."
    )

st.caption(
    "Note: API file counts may include overlapping records. "
    "The charts describe the collected sample and do not establish "
    "causal relationships or guarantee model performance."
)