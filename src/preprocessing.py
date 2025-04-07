import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# Load the iris dataset from CSV
data = pd.read_csv('data/iris.csv')
X = data.drop('target', axis=1)
y = data['target']

# Create preprocessing pipeline
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Apply preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_include=["float64", "int64"]))
    ]
)

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

# Convert back to dataframe for easier handling
X_processed_df = pd.DataFrame(
    X_processed,
    columns=X.columns,
    index=X.index
)

# Add target back to processed data
processed_data = X_processed_df.copy()
processed_data['target'] = y

# Save processed data
processed_data.to_csv('data/processed_iris.csv', index=False)

print(f"Original data shape: {X.shape}")
print(f"Processed data shape: {X_processed.shape}")
print("Preprocessing complete!")
