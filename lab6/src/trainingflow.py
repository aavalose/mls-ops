from metaflow import FlowSpec, step
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import mlflow
import mlflow.sklearn


class IrisTrainingFlow(FlowSpec):

    @step
    def start(self):
        """Load Iris dataset and split into train/test"""
        data = load_iris()
        X = data.data
        y = data.target
        self.feature_names = data.feature_names

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.next(self.train_knn, self.train_svm)

    @step
    def train_knn(self):
        """Train KNN classifier"""
        self.model_name = "KNN"
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        self.next(self.choose_model)

    @step
    def train_svm(self):
        """Train SVM classifier"""
        self.model_name = "SVM"
        self.model = SVC(kernel="linear")
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        """Compare models and log best one with MLflow"""
        import operator

        best_input = max(inputs, key=operator.attrgetter("score"))
        self.model = best_input.model
        self.model_name = best_input.model_name
        self.score = best_input.score
        self.X_test = best_input.X_test
        self.y_test = best_input.y_test

        # MLflow logging (local server assumed)
        mlflow.set_tracking_uri("file:mlruns")
        mlflow.set_experiment("iris-metaflow-experiment")

        with mlflow.start_run():
            mlflow.log_param("model_type", self.model_name)
            mlflow.log_metric("accuracy", self.score)
            mlflow.sklearn.log_model(self.model, "iris_model")
        self.next(self.end)

    @step
    def end(self):
        """Done"""
        print(f"Best model: {self.model_name} with accuracy: {self.score:.3f}")


if __name__ == "__main__":
    IrisTrainingFlow()