import pandas as pd

# Load the parquet file
df = pd.read_parquet("train-00000-of-00001-6815e36c7337e2db.parquet")

# Print available columns to check if 'response' is present
print("Columns in DataFrame:", df.columns.tolist())
