import pandas as pd

# List of parquet files
parquet_files = [
    "textbook-quality-programming-train-00000-of-00001-6815e36c7337e2db.parquet",
    "textbooks-are-all-you-need-lite_train-00000-of-00007-eb50287110fc883d.parquet",
    "tiny-codes-part_1_200000.parquet",
    "tiny-code-textbooks-part_10_100000.parquet",
]

# Loop through each parquet file, load the DataFrame, and print the columns
for file in parquet_files:
    df = pd.read_parquet(file)
    print(f"Columns in {file}:")
    print(df.columns)
    print("\n")
