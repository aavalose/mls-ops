# Use the official Python image
FROM python:3.10-slim

# Set environment variables
ENV MLFLOW_HOME=/mlflow

# Set working directory
WORKDIR $MLFLOW_HOME

# Install MLflow and any dependencies
RUN pip install --no-cache-dir mlflow

# Expose the port MLflow uses
EXPOSE 5000

# Run MLflow server when the container starts
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]