import pandas as pd
import re
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename="process_log.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# File paths and their respective columns of interest
files_and_columns = {
    "textbook-quality-programming-train-00000-of-00001-6815e36c7337e2db.parquet": "markdown",
    "textbooks-are-all-you-need-lite_train-00000-of-00007-eb50287110fc883d.parquet": "completion",
    "tiny-codes-part_1_200000.parquet": "response",
    "tiny-code-textbooks-part_10_100000.parquet": "content",
}

# Load the list of words from the text file
with open("ACT_test_words_unique.txt", "r") as file:
    word_list = file.read().split()

# Prepare a regex pattern for matching words, ignoring case
pattern = r"\b(" + "|".join(re.escape(word) for word in word_list) + r")\b"
regex = re.compile(pattern, re.IGNORECASE)
for file_path, text_column in files_and_columns.items():
    df = pd.read_parquet(file_path).sample(20)  # Load and sample 20 random rows
    match_percentages = []

    for index, row in df.iterrows():
        text = row[text_column]
        matches = regex.findall(text)
        match_count = len(matches)
        total_words = len(text.split())
        match_percent = (match_count / total_words) * 100 if total_words else 0
        match_percentages.append(match_percent)

        logging.info(
            f"File: {file_path}, Index: {index}, Match Percentage: {match_percent:.2f}%"
        )

    # Log the mean match percentage for the sampled data
    mean_match_percentage = np.mean(match_percentages)
    logging.info(f"Mean Match Percentage for {file_path}: {mean_match_percentage:.2f}%")

    # Output the first 5 rows of the sampled data to a CSV file
    output_csv_path = f"sample_output_{file_path.replace('.parquet', '')}.csv"
    df.head(5).to_csv(output_csv_path, index=False)
    logging.info(f"Sampled data written to {output_csv_path}")
