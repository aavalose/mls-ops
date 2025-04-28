# lab8app/retrain_and_log_model.py

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate a quick synthetic dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)

# Start a new MLflow run
with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X, y)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Print the Run ID for your FastAPI app
    print("✅ Model logged successfully!")
    print(f"Run ID: {run.info.run_id}")
    print(f"Model URI: runs:/{run.info.run_id}/model")