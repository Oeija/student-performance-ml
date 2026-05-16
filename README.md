# Student Performance ML

An end-to-end machine learning pipeline to predict student math score based on demographic, socioeconomic, and academic features.

**Frontend:** [student-performance-frontend](https://github.com/Oeija/student-performance-frontend)

## Project Overview

This project predicts a student's math score using features such as:
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

The pipeline includes:
- **Data ingestion**: Load raw CSV and split into train/test sets
- **Data validation**: Comprehensive data quality checks (schema, nulls, types, ranges, categoricals, duplicates)
- **Data transformation**: Feature engineering and preprocessing
- **Model training**: Hyperparameter tuning across multiple algorithms
- **SHAP-based model interpretability**: Feature importance explanations
- **FastAPI JSON API**: Real-time predictions with input validation

## Tech Stack

- **ML / Data:** [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [XGBoost](https://xgboost.readthedocs.io/), [CatBoost](https://catboost.ai/)
- **Explainability:** [SHAP](https://shap.readthedocs.io/)
- **API:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Pydantic](https://docs.pydantic.dev/)
- **Visualization:** [seaborn](https://seaborn.pydata.org/), [matplotlib](https://matplotlib.org/), [plotly](https://plotly.com/python/)
- **Serialization:** [dill](https://dill.readthedocs.io/)
- **Configuration:** [PyYAML](https://pyyaml.org/)
- **Testing:** [pytest](https://docs.pytest.org/)
- **Deployment:** Docker on [AWS EC2](https://aws.amazon.com/ec2/)

## Architecture

```
config/
├── config.yaml                    # Centralized configuration
data/
├── raw/
│   └── StudentsPerformance.csv    # Source data (version controlled)
├── processed/                     # Generated datasets (NOT version controlled)
│   ├── train.csv
│   └── test.csv
artifacts/                         # Generated model artifacts (gitignored, baked into Docker image)
├── model.pkl
├── preprocessor.pkl
├── feature_names.json
├── shap_summary.png
└── shap_importance.csv
src/
├── components/
│   ├── data_ingestion.py          # Data loading and train-test splitting
│   ├── data_validation.py         # Data quality validation (nulls, types, ranges, categoricals, duplicates)
│   ├── data_transformation.py     # Feature engineering and preprocessing
│   ├── model_trainer.py           # Model training and evaluation
│   └── model_interpretability.py  # SHAP explanations
├── pipeline/
│   └── predict_pipeline.py        # Prediction pipeline
├── config.py                      # Pydantic configuration schema
├── utils.py                       # Utility functions
├── exception.py                   # Custom exceptions
└── logger.py                      # Logging configuration
app.py                             # FastAPI application
Dockerfile                         # Container image for deployment
pyproject.toml                     # Project dependencies
requirements.txt                   # Auto-generated dependencies
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd student-performance-ml
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -e .
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

### 4. Run the training pipeline

```bash
python src/components/data_ingestion.py
```

This will:
- Load raw data
- Validate data quality via `DataValidation` component
- Preprocess features
- Train multiple models with hyperparameter tuning
- Select the best model (currently Ridge regression)
- Generate SHAP interpretability plots in `artifacts/`

### 5. Start the FastAPI Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`

Interactive API documentation:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated list of allowed frontend origins |

## API Endpoints

### Health Check
- **GET** `/health`
- Returns server and model load status

### Model Info
- **GET** `/api/model-info`
- Returns model metadata and SHAP feature importance rankings

### Predict Math Score
- **POST** `/api/predict`
- Submit student data to get math score prediction
- Request body:
  ```json
  {
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "none",
    "reading_score": 72,
    "writing_score": 74
  }
  ```
- Response:
  ```json
  {
    "prediction": 73,
    "status": "success"
  }
  ```

### API Documentation
- **GET** `/docs` — Swagger UI (interactive API docs)
- **GET** `/redoc` — ReDoc documentation

## Configuration

All project settings are centralized in `config/config.yaml`. Key settings include:

- `data_ingestion.test_size`: Train-test split ratio (default: 0.2)
- `model_trainer.cv_folds`: Cross-validation folds (default: 5)
- `model_trainer.min_score_threshold`: Minimum R² threshold (default: 0.6)

## Data Validation

The `DataValidation` component (`src/components/data_validation.py`) performs comprehensive data quality checks before any model training occurs. All validations follow a **fail-fast** approach — if any check fails, the pipeline stops immediately with a descriptive error.

Validation checks include:

| Check | Description | Fail Condition |
|-------|-------------|----------------|
| **Schema Validation** | Ensures all expected columns are present | Missing required columns |
| **Null Detection** | Rejects any `NaN` or `null` values | Any null value in any column |
| **Data Type Enforcement** | Strict type checking on all columns | Score columns must be `int64`; categorical columns must be `object`/`string` |
| **Score Range Validation** | Validates numerical score bounds | Any score outside [0, 100] |
| **Categorical Value Validation** | Checks categorical columns against allowed enums | Invalid values in `gender`, `race_ethnicity`, `lunch`, etc. |
| **Empty String / Placeholder Detection** | Rejects empty, whitespace-only, or placeholder strings | `""`, `"   "`, `"N/A"`, `"null"`, `"-"`, `"?"`, `"unknown"`, `"None"` |
| **Duplicate Removal** | Drops exact duplicate rows | Logs count of removed duplicates |

These validations ensure data integrity and prevent silent failures during training or prediction.

## Model Performance

Current best model: **Ridge Regression**
- R² Score: ~0.88
- The model uses L2 regularization with tuned alpha parameter

## Model Interpretability

SHAP (SHapley Additive exPlanations) is used to explain model predictions:

- **Global interpretability**: `artifacts/shap_summary.png` shows feature importance across the dataset
- **Feature importance**: `artifacts/shap_importance.csv` ranks features by their impact

## Testing

Run unit tests:

```bash
pytest tests/
```

Test coverage includes:
- `tests/test_data_validation.py` — Data quality validation checks
- `tests/test_data_ingestion.py` — File loading and train-test splitting
- `tests/test_data_transformation.py` — Feature preprocessing
- `tests/test_model_trainer.py` — Model training and evaluation
- `tests/test_config.py` — Configuration loading and validation

## Docker Deployment (AWS EC2)

### Prerequisites
- Docker installed locally and on EC2
- Model artifacts (`artifacts/`) present locally before building

### Build Image

```bash
docker build -t yourusername/student-performance-ml:1.0.0 .
```

### Push to Docker Hub

```bash
docker login
docker tag student-performance-ml:1.0.0 yourusername/student-performance-ml:1.0.0
docker push yourusername/student-performance-ml:1.0.0
```

### Run on AWS EC2

```bash
docker pull yourusername/student-performance-ml:1.0.0

docker run -d \
  --name student-performance-ml \
  -p 8000:8000 \
  --restart unless-stopped \
  -e CORS_ORIGINS="http://localhost:3000,https://your-frontend.vercel.app" \
  yourusername/student-performance-ml:1.0.0
```

The API will be available at `http://your-ec2-instance:8000`.

## Important Notes

- **Artifacts**: The `artifacts/` directory is listed in `.gitignore` and `.dockerignore`, but **must exist locally** before building the Docker image. The `COPY . .` instruction in the Dockerfile includes these files in the final image.
- **Working Directory**: The application resolves artifact paths relative to the project root. The Dockerfile sets `WORKDIR /app` to ensure correct resolution.
- **CORS**: Update the `CORS_ORIGINS` environment variable whenever you deploy a new frontend domain.

## Data Source

Dataset: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

## Author

Vincent Oei (oei.vincent20@gmail.com)
