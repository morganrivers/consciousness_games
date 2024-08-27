import pandas as pd
import re

# Load the parquet file
df = pd.read_parquet("train-00000-of-00001-6815e36c7337e2db.parquet")

# Load the list of words from the text file
with open("ACT_test_common_words_removed_chatbot_additions.txt", "r") as file:
    word_list = file.read().split()

# Prepare a regex pattern for matching words, ignoring case
pattern = r"\b(" + "|".join(re.escape(word) for word in word_list) + r")\b"
regex = re.compile(pattern, re.IGNORECASE)

# Process the first 1000 rows
results = []
for index, row in df.head(50).iterrows():
    print(f"Processing index {index}.")
    # Updated to use 'markdown' column
    matches = regex.findall(row["markdown"])
    match_count = len(matches)
    total_words = len(row["markdown"].split())
    # print("matches")
    # print(matches)
    # print("match_count")
    # print(match_count)
    # print("total_words")
    # print(total_words)
    match_percent = (match_count / total_words) * 100 if total_words else 0

    results.append(
        {
            "Index": index,
            "Match Percentage": f"{match_percent:.2f}%",
            "Matching Words": set(matches),
            "Response Preview": f"{row['markdown'][:200]}...{row['markdown'][-100:]}",  # Updated here as well
        }
    )

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Sort the DataFrame by 'Match Percentage'
results_df["Match Percentage"] = (
    results_df["Match Percentage"].str.rstrip("%").astype(float)
)
results_df = results_df.sort_values(by="Match Percentage", ascending=True)

# Select the 50 least and 50 most matching rows
least_matching = results_df.head(10)
most_matching = results_df.tail(10)

# Print the 50 least and 50 most matching rows
print("50 Least Matching Rows:")
print(least_matching)
print("\n50 Most Matching Rows:")
print(most_matching)

# Optionally, save these subsets to CSV files
least_matching.to_csv("least_matching_results.csv", index=False)
most_matching.to_csv("most_matching_results.csv", index=False)
