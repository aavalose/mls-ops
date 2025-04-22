from metaflow import FlowSpec, step
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris

class IrisScoringFlow(FlowSpec):

    @step
    def start(self):
        """Prepare input data for scoring (simulate new data)"""
        iris = load_iris()
        self.X_input = iris.data[:5]  # Simulate new unseen examples
        self.next(self.load_model)

    @step
    def load_model(self):
        """Load the latest trained model from MLflow"""
        # Set the correct absolute tracking URI path
        mlflow.set_tracking_uri("file:/Users/arturoavalos/Documents/MSDS/4mod/mlops/lab6/src/mlruns")

        # Specify the correct run ID
        run_id = "63c44790db5344f881022bad034c2136"

        # Construct the full model URI
        model_uri = f"runs:/{run_id}/iris_model"

        # Load the model
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        """Make predictions using the loaded model"""
        self.predictions = self.model.predict(self.X_input)
        print("Predictions on new data:", self.predictions)
        self.next(self.end)

    @step
    def end(self):
        """Done"""
        print("Scoring complete.")

if __name__ == "__main__":
    IrisScoringFlow()