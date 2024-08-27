import pandas as pd
import re
import logging

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

# Load the parquet file
# df = pd.read_parquet("tiny-code-textbooks-part_10_100000.parquet")
df = pd.read_parquet(
    "textbooks-are-all-you-need-lite_train-00000-of-00007-eb50287110fc883d.parquet"
)

# Define column containing text based on the dataset
# Update this line to dynamically select the correct column name based on your datasets
# text_column = (
#     "content","response"  # Change to 'response', 'content', 'prompt', etc. based on dataset
# )

# Load the list of words from the text file
with open("ACT_test_words_unique.txt", "r") as file:
    word_list = file.read().split()

# Prepare a regex pattern for matching words, ignoring case
pattern = r"\b(" + "|".join(re.escape(word) for word in word_list) + r")\b"
regex = re.compile(pattern, re.IGNORECASE)

# Process all rows
results = []
for index, row in df.head(1000).iterrows():
    text = (
        "\nCOMPLETION: "
        + row["completion"]
        + "\nFIRST_TASK: "
        + row["first_task"]
        + "\nSECOND_TASK: "
        + row["second_task"]
        + "\nLAST_TASK: "
        + row["last_task"]
    )  # Update this to use the dynamic column
    logging.info(f"Processing index {index}.")
    matches = regex.findall(text)
    match_count = len(matches)
    total_words = len(text.split())
    match_percent = (match_count / total_words) * 100 if total_words else 0

    results.append(
        {
            "Index": index,
            "Match Percentage": f"{match_percent:.2f}%",
            "Matching Words": set(matches),
            # "Response Preview": f"{text[:200]}...{text[-100:]}",
            "Response Preview": text,
        }
    )

# Convert results to a DataFrame and sort by 'Match Percentage'
results_df = pd.DataFrame(results)
results_df["Match Percentage"] = (
    results_df["Match Percentage"].str.rstrip("%").astype(float)
)
results_df = results_df.sort_values(by="Match Percentage", ascending=True)

# Optionally, save results to CSV file
results_df.to_csv("matching_results.csv", index=False)

# Log some results for quick viewing
logging.info(
    "Sample Matching Rows:\n%s", results_df.head(20)
)  # Adjust the head() value as needed for different logs
