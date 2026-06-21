import pandas as pd

df = pd.read_csv("data/sample_submission.csv")

print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)