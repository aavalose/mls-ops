from sklearn import datasets
import pandas as pd

# Load the iris dataset
data = datasets.load_iris(as_frame=True)
df = data.data
df['target'] = data.target

# Save to csv
df.to_csv('data/iris.csv', index=False)
print("Iris dataset saved to data/iris.csv")
