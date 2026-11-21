# InterviewAce AI

### Domain-Specific Interview Preparation Using RAG, LLMs, Multi-API Data Fusion, and Machine Learning

## 1. Project Overview

InterviewAce AI is a domain-specific interview preparation application that combines external technology data sources, data fusion, machine learning, and Retrieval-Augmented Generation (RAG).

The project collects information from five external APIs, consolidates the collected records, performs exploratory data analysis and data-quality checks, evaluates machine learning models, and integrates the fused data into a retrieval system to provide context-aware answers to user queries.

## 2. Objectives

* Collect relevant information from multiple external APIs.
* Combine data from different sources into structured datasets.
* Assess data quality and perform exploratory data analysis.
* Compare source-level data with fused datasets.
* Generate data-driven insights.
* Evaluate regression and classification algorithms.
* Integrate fused information into a RAG-based question-answering system.
* Compare and document the strengths and limitations of the implemented methods.

## 3. External APIs

The project uses the following five data sources.

| API / Source   | Data Collected                                                               | Purpose                                                   |
| -------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------- |
| GitHub         | Repository information for Python, SQL, machine learning, and data analysis  | Explore technology repositories and development activity  |
| Stack Exchange | Questions and related metadata for Python, SQL, pandas, and machine learning | Obtain technical questions and community-interaction data |
| Remotive       | Remote job listings across selected technology categories                    | Incorporate job-related information                       |
| Open Library   | Book metadata                                                                | Add book and learning-resource information                |
| Hacker News    | Technology-related discussion records                                        | Include technology-community content                      |

The collected data is stored as CSV files in `data/api_data/`.

## 4. Data Fusion

The project uses two complementary forms of data consolidation.

### 4.1 Horizontal Fusion

File: `data/fusion/InterviewAce_horizontal_fusion.csv`

This dataset aggregates selected metrics by technology category. It supports comparisons across categories and exploratory analysis.

The associated Excel workbook is:

`data/fusion/InterviewAce_API_Data_Fusion.xlsx`

### 4.2 Document-Level Fusion

File: `data/InterviewAce_Data_Fusion.xlsx`

This workbook consolidates records from all five sources into a common structure containing the source name, original filename, and content.

This representation supports cross-source retrieval and integration with the RAG pipeline.

The two fusion datasets serve different purposes: horizontal fusion supports category-level analysis, while document-level fusion preserves source records for retrieval.

## 5. Data Processing and Analysis

The `data_processing/` directory contains the project's processing scripts.

| Script                       | Purpose                                      |
| ---------------------------- | -------------------------------------------- |
| `api_data_loader.py`         | Loads API datasets                           |
| `clean_api_data.py`          | Cleans collected API data                    |
| `build_horizontal_fusion.py` | Creates the horizontal fusion dataset        |
| `fusion.py`                  | Performs data-fusion operations              |
| `data_quality_analysis.py`   | Assesses data quality                        |
| `validate_fusion.py`         | Validates the fusion output                  |
| `eda_fused_data.py`          | Performs exploratory data analysis           |
| `before_after_fusion.py`     | Compares source-level and fused data         |
| `generate_insights.py`       | Generates analytical insights                |
| `ml_experiment.py`           | Trains and evaluates machine learning models |
| `ml_visualizations.py`       | Generates ML evaluation charts               |
| `build_api_vectorstore.py`   | Builds the API-data retrieval index          |

## 6. Exploratory Data Analysis

The EDA process examines the fused dataset and generates visualizations to understand the data distribution and relationships.

EDA outputs are stored in:

`data/eda/`

The comparison outputs are stored in:

`data/comparison/`

These outputs include source-record comparisons, fusion summaries, and technology-popularity visualizations.

## 7. Data Quality

The data-quality analysis evaluates the collected and fused data for potential quality issues.

Report:

`data/quality/InterviewAce_Data_Quality_Report.xlsx`

The report supports the identification and documentation of data-quality concerns before downstream analysis.

## 8. Machine Learning Experiments

The project evaluates five algorithms using Stack Exchange Python question data.

### Regression Models

1. Linear Regression
2. Random Forest Regressor
3. Decision Tree Regressor
4. Gradient Boosting Regressor

Regression metrics include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).

### Classification Model

5. Random Forest Classifier

The classifier predicts low, medium, or high score categories. The categories are created from score quantiles.

Evaluation results are saved in:

* `data/ml_results/model_evaluation.csv`
* `data/ml_results/classification_evaluation.csv`

ML evaluation charts are stored in `data/eda/`.

### Interpretation

The current results indicate limited predictive performance. The Decision Tree Regressor achieves the highest R² among the four regression models, while Gradient Boosting achieves the lowest MAE.

The Random Forest Classifier achieves 35% accuracy on the test set.

These findings are preliminary and should not be interpreted as evidence that the models predict interview success or job-market demand. The regression target is the Stack Exchange post score.

## 9. RAG and LLM Integration

The project integrates the fused API dataset into a retrieval system.

The retrieval pipeline uses document embeddings and a FAISS vector index to retrieve relevant context for user queries. Retrieved information is used alongside an LLM to generate answers.

The API-data vector index is stored in:

`vector_db_api/`

The application also supports interview-preparation questions based on its existing document-processing and retrieval pipeline.

The RAG implementation is designed to support both general technical interview questions and relevant queries against the collected API data.

## 10. Project Structure

```text
InterviewAce-AI/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   ├── api_data/
│   ├── cleaned/
│   ├── comparison/
│   ├── eda/
│   ├── fusion/
│   ├── insights/
│   ├── ml_ready/
│   ├── ml_results/
│   ├── quality/
│   └── InterviewAce_Data_Fusion.xlsx
├── data_processing/
│   ├── api_data_loader.py
│   ├── before_after_fusion.py
│   ├── build_api_vectorstore.py
│   ├── build_horizontal_fusion.py
│   ├── clean_api_data.py
│   ├── data_quality_analysis.py
│   ├── eda_fused_data.py
│   ├── fusion.py
│   ├── generate_insights.py
│   ├── ml_experiment.py
│   ├── ml_visualizations.py
│   └── validate_fusion.py
├── utils/
└── vector_db_api/
```

## 11. Environment Setup

Use Python 3.11 in the development environment.

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The application also requires the configured LLM runtime and model if using the local Ollama-based generation pipeline.

## 12. Running the Application

From the project root, run:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal.

## 13. Reproducing the ML Experiments

Run the ML experiment:

```powershell
python data_processing/ml_experiment.py
```

Generate evaluation visualizations:

```powershell
python data_processing/ml_visualizations.py
```

The scripts save their evaluation results and charts in the project data directories.

## 14. Limitations

* The datasets cover selected technology categories and API responses rather than the entire technology job market.
* Different APIs provide different fields and record structures.
* Some API records may be duplicated or incomplete.
* Category-level popularity measures are constructed analytical indicators, not validated measures of real-world popularity.
* The ML experiments use a limited sample of 500 Stack Exchange Python records.
* The regression models show limited predictive performance.
* The classifier predicts score-based categories, not interview outcomes.
* Job-role matching in the retrieval pipeline may use broad category-based matching rather than verified skill requirements.
* API availability, response formats, and rate limits may affect reproducibility.

## 15. Conclusion

InterviewAce AI demonstrates a combined workflow for collecting multi-source technology data, performing data fusion and analysis, evaluating machine learning algorithms, and integrating the consolidated data into a RAG-based interview-preparation application.

The project provides a foundation for further improvements in data quality, retrieval relevance, model evaluation, and evidence-based technical interview preparation.
