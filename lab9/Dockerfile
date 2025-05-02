# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY redditApp.py .
COPY reddit_model_pipeline.joblib .

# Install dependencies
RUN pip install fastapi uvicorn scikit-learn joblib pydantic

# Run the app
CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8080"]