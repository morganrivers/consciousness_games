import json
import re
import logging
import pandas as pd

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


# Load JSONL data
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line.strip()) for line in file]


# Load the list of words from the text file
with open("ACT_test_words_unique.txt", "r") as file:
    word_list = file.read().split()

# Prepare a regex pattern for matching words, ignoring case
pattern = r"\b(" + "|".join(re.escape(word) for word in word_list) + r")\b"
regex = re.compile(pattern, re.IGNORECASE)

# Load JSONL data
data = load_data(
    "python/final/jsonl/test/python_test_0.jsonl"
)  # Provide the path to your JSONL file

# Process all rows
results = []

# Process up to the first 1000 rows
max_items = 10000  # Set a limit for the number of items to process
results = []
for index, item in enumerate(
    data[:max_items]
):  # Slicing the data list to handle only the first 1000 items
    text = item[
        "original_string"
    ]  # Assumes 'original_string' contains the relevant text
    url = item["url"]  # Log or use URL as needed
    logging.info(f"Processing index {index}, URL: {url}.")
    matches = regex.findall(text)
    match_count = len(matches)
    total_words = len(text.split())
    match_percent = (match_count / total_words) * 100 if total_words else 0

    results.append(
        {
            "Index": index,
            "URL": url,
            "Match Percentage": f"{match_percent:.2f}%",
            "Matching Words": set(matches),
            "Response Preview": f"{text[:200]}...{text[-100:]}",  # Truncate long texts for preview
        }
    )

# Convert results to a DataFrame and sort by 'Match Percentage'
results_df = pd.DataFrame(results)
results_df["Match Percentage"] = (
    results_df["Match Percentage"].str.rstrip("%").astype(float)
)
results_df = results_df.sort_values(by="Match Percentage", ascending=False)

# Optionally, save results to CSV file
results_df.to_csv("matching_results.csv", index=False)

# Log some results for quick viewing
logging.info(
    "Top Matching Rows:\n%s", results_df.head(10).to_string()
)  # Show top 10 matches
