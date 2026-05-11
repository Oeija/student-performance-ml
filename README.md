# Student Performance ML

An end-to-end machine learning pipeline to predict student math scores based on demographic and academic features.

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
- Data ingestion with validation
- Data transformation and preprocessing
- Model training with hyperparameter tuning
- SHAP-based model interpretability
- FastAPI JSON API for predictions

## Architecture

```
config/
├── config.yaml              # Centralized configuration
data/
├── raw/
│   └── StudentsPerformance.csv    # Source data (version controlled)
├── processed/                     # Generated datasets (NOT version controlled)
│   ├── train.csv
│   └── test.csv
artifacts/                         # Generated model artifacts (NOT version controlled)
├── model.pkl
├── preprocessor.pkl
├── feature_names.json
├── shap_summary.png
└── shap_importance.csv
src/
├── components/
│   ├── data_ingestion.py    # Data loading and validation
│   ├── data_transformation.py # Feature engineering
│   ├── model_trainer.py     # Model training and evaluation
│   └── model_interpretability.py # SHAP explanations
├── pipeline/
│   ├── predict_pipeline.py  # Prediction pipeline
│   └── train_pipeline.py    # Training pipeline
├── config.py                # Pydantic configuration schema
├── utils.py                 # Utility functions
├── exception.py             # Custom exceptions
└── logger.py                # Logging configuration
app.py                       # FastAPI application
Dockerfile                   # Container image for AWS deployment
pyproject.toml               # Project dependencies
requirements.txt             # Auto-generated dependencies
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
- Load and validate the raw data
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

## Configuration

All project settings are centralized in `config/config.yaml`. Key settings include:

- `data_ingestion.test_size`: Train-test split ratio (default: 0.2)
- `model_trainer.cv_folds`: Cross-validation folds (default: 5)
- `model_trainer.min_score_threshold`: Minimum R² threshold (default: 0.6)

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

## Docker Deployment (AWS)

Build the Docker image:

```bash
docker build -t student-performance-ml:1.0.0 .
```

Run the container:

```bash
docker run --name student-performance-ml -d -p 8000:8000 student-performance-ml:1.0.0
```

The API will be available at `http://localhost:8000`.

> **Note:** The default `CORS_ORIGINS` allows `http://localhost:3000`. To allow a deployed frontend, pass the `--env` flag:
> ```bash
> docker run --name student-performance-ml -d -p 8000:8000 --env CORS_ORIGINS=https://your-frontend-link student-performance-ml:1.0.0
> ```

For AWS deployment (ECS, EKS, or App Runner):
1. Push the image to Amazon ECR
2. Configure the `CORS_ORIGINS` environment variable with your frontend URL
3. Expose port 8000

## CI/CD

This project is configured for CI/CD with:
- `pyproject.toml` for dependency management
- `requirements.txt` for deployment environments
- Unit tests in `tests/`
- Dockerfile for containerized AWS deployment

## Data Source

Dataset: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

## Author

Vincent Oei (oei.vincent20@gmail.com)
