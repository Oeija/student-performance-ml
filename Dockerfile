FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by scikit-learn, xgboost, shap, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for better caching)
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

# Expose the application port
EXPOSE 8000

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
