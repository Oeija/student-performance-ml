import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# ---------------------------------------------------------------------------
# Load SHAP feature importance at startup
# ---------------------------------------------------------------------------
_feature_importance: List[dict] = []
try:
    import csv

    with open(CONFIG.model_evaluation.shap_importance_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _feature_importance.append(
                {
                    "feature": row["feature"],
                    "mean_abs_shap": float(row["mean_abs_shap"]),
                }
            )
except Exception as e:
    logging.warning(f"Could not load SHAP feature importance: {e}")

_predict_pipeline: PredictPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predict_pipeline
    try:
        _predict_pipeline = PredictPipeline()
        logging.info("PredictPipeline initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize PredictPipeline: {e}")
    yield
    logging.info("Shutting down API")


app = FastAPI(
    title="Student Performance ML API",
    description="Predict student math scores based on demographic, socioeconomic, and academic features",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in CONFIG.validations.valid_genders:
            raise ValueError(
                f"Invalid gender. Must be one of: {CONFIG.validations.valid_genders}"
            )
        return v

    @field_validator("race_ethnicity")
    @classmethod
    def validate_ethnicity(cls, v: str) -> str:
        if v not in CONFIG.validations.valid_race_ethnicity:
            raise ValueError(
                f"Invalid ethnicity. Must be one of: {CONFIG.validations.valid_race_ethnicity}"
            )
        return v

    @field_validator("parental_level_of_education")
    @classmethod
    def validate_education(cls, v: str) -> str:
        if v not in CONFIG.validations.valid_parental_education:
            raise ValueError(
                f"Invalid parental education. Must be one of: {CONFIG.validations.valid_parental_education}"
            )
        return v

    @field_validator("lunch")
    @classmethod
    def validate_lunch(cls, v: str) -> str:
        if v not in CONFIG.validations.valid_lunch:
            raise ValueError(
                f"Invalid lunch type. Must be one of: {CONFIG.validations.valid_lunch}"
            )
        return v

    @field_validator("test_preparation_course")
    @classmethod
    def validate_test_prep(cls, v: str) -> str:
        if v not in CONFIG.validations.valid_test_preparation:
            raise ValueError(
                f"Invalid test preparation. Must be one of: {CONFIG.validations.valid_test_preparation}"
            )
        return v

    @field_validator("reading_score", "writing_score")
    @classmethod
    def validate_scores(cls, v: float) -> float:
        if not (CONFIG.validations.score_min <= v <= CONFIG.validations.score_max):
            raise ValueError(
                f"Score must be between {CONFIG.validations.score_min} and {CONFIG.validations.score_max}"
            )
        return v


class PredictionResponse(BaseModel):
    prediction: int
    status: str = "success"


class FeatureImportanceItem(BaseModel):
    feature: str
    mean_abs_shap: float


class ModelInfoResponse(BaseModel):
    model_type: str
    r2_score: float
    cv_folds: int
    feature_importance: List[FeatureImportanceItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Student Performance ML API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=_predict_pipeline is not None,
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_type="Ridge Regression",
        r2_score=0.88,
        cv_folds=CONFIG.model_trainer.cv_folds,
        feature_importance=[
            FeatureImportanceItem(**item) for item in _feature_importance
        ],
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if _predict_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )

    try:
        data = CustomData(
            gender=request.gender,
            race_ethnicity=request.race_ethnicity,
            parental_level_of_education=request.parental_level_of_education,
            lunch=request.lunch,
            test_preparation_course=request.test_preparation_course,
            reading_score=int(request.reading_score),
            writing_score=int(request.writing_score),
        )
        pred_df = data.get_data_as_data_frame()
        results = _predict_pipeline.predict(pred_df)
        return PredictionResponse(prediction=results[0])

    except CustomException as e:
        logging.error(f"Prediction pipeline error: {e.error_message}")
        raise HTTPException(
            status_code=400,
            detail="Prediction failed due to invalid input or processing error.",
        )
    except Exception as e:
        logging.error(f"Unexpected prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction.",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
