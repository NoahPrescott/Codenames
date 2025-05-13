import json
import numpy as np
from collections import defaultdict

file_paths = [
    # gpt clues
    # "gpt-clue-gpt-relatedness-oct20batch-1.json",
    # "gpt-clue-gpt-relatedness-oct20batch-2.json",
    # "gpt-clue-gpt-relatedness-oct20batch-3.json",
    # "gpt-clue-gpt-relatedness-oct20batch-4.json"

    #human clues
    "human-clue-gpt-relatedness-oct10batch-1.json",
    "human-clue-gpt-relatedness-oct10batch-2.json",
    "human-clue-gpt-relatedness-oct10batch-3.json",
    "human-clue-gpt-relatedness-oct10batch-4.json"
]

data = defaultdict(lambda: defaultdict(list))
judgment_sources = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # To store judgments per file

# Counters for new statistics
unanimous_count = 0
total_judgments = 0  # Assuming each file has the same number of judgments
large_difference_count = 0

# Step 1: Load JSON Files and Store Judgments per File
for file_index, file_path in enumerate(file_paths):
    with open(file_path, "r") as file:
        json_data = json.load(file)
        for clue, words in json_data.items():
            for word, judgment in words.items():
                judgment_value = int(judgment)
                data[clue][word].append(judgment_value)  # Store the judgment
                judgment_sources[clue][word][file_path].append(judgment_value)  # Track file source

# Calculate total judgments count
total_judgments = sum(len(words) for words in data.values())

# Step 2: Compute Averages and Variances
averaged_data = {}
variance_data = {}

for clue, words in data.items():
    averaged_data[clue] = {}
    variance_data[clue] = {}
    for word, judgments in words.items():
        # Calculate average and variance
        avg_judgment = np.mean(judgments)
        variance_judgment = np.var(judgments)
        averaged_data[clue][word] = round(avg_judgment, 2)
        variance_data[clue][word] = round(variance_judgment, 2)
        
        # Check for unanimous judgments
        if len(set(judgments)) == 1:  # All judgments are identical
            unanimous_count += 1

        # Check for large differences between any two files' judgments
        for i in range(len(judgments)):
            for j in range(i + 1, len(judgments)):
                if abs(judgments[i] - judgments[j]) > 20:
                    large_difference_count += 1
                    break  # Only count once per pair

# Save the averaged judgments to a new JSON file
with open("human-clue-gpt-relatedness-oct10-averaged-judgments.json", "w") as avg_file:
    json.dump(averaged_data, avg_file, indent=4)

# Step 3: Print New Statistics
print("Summary of Judgments Across Files:\n")
print(f"Total Judgments in Each File: {total_judgments}")
print(f"Unanimous Judgments Across All Files: {unanimous_count}")
print(f"Judgments with >20 Difference Between Any Two Files: {large_difference_count}")

# Step 4: Find the Top 5 Most Divergent Clue-Word Pairs (for completeness)
divergences = []

for clue, words in data.items():
    for word, judgments in words.items():
        max_diff = max(judgments) - min(judgments)  # Calculate divergence
        divergences.append((clue, word, max_diff, judgments))  # Store clue, word, divergence, and judgments

# Sort by divergence and take the top 5
top_5_divergent = sorted(divergences, key=lambda x: x[2], reverse=True)[:5]

print("\nTop 5 Most Divergent Clue-Word Pairs:\n")
for clue, word, divergence, judgments in top_5_divergent:
    print(f"Clue: {clue}, Word: {word}")
    print(f" Divergence: {divergence}")
    print(f" Judgments: {judgments}")
    print(" Appears in files:")
    
    # Print judgments from each file
    for file_path, judgment_list in judgment_sources[clue][word].items():
        print(f"  - {file_path}: Judgments = {judgment_list}")
    print("\n")  # Add spacing between entries

# Calculate and print overall statistics
total_variance = [np.var(judgments) for _, _, _, judgments in divergences]
average_variance = np.mean(total_variance)
overall_max_divergence = top_5_divergent[0][2]  # Highest divergence in the sorted list

print(f"\nAverage Variance across all judgments: {average_variance:.2f}")
print(f"Maximum Divergence (most differing judgment): {overall_max_divergence}")
