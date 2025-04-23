#!/bin/bash

echo "Starting MLflow setup..."

POSTGRESQL_URL="postgresql+psycopg2://mlflowuser:${MLFLOW_DB_PASSWORD}@/mlflowdb?host=/cloudsql/msds-mlops-2025:us-west2:mlflow-sql-instance"

echo "Upgrading DB..."
mlflow db upgrade "$POSTGRESQL_URL"

echo "Starting server..."
mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri "$POSTGRESQL_URL" \
  --artifacts-destination gs://mlflow-artifacts-2025/mlruns