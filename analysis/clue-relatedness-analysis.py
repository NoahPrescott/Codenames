# utilizes superdata!.json and relatedness-data.json
# superdata!.json contains each board along with GPT & human clues, GPT & human intended words, and GPT & human guesses
# relatedness-data.json each each judgment comparison as the two words compared, the comparer, and the relatedness score
# relatedness-data.json currently refers to a 'clue' and a 'word'; this is vestigial. These equate to word1 and word2. 

import json
import statistics
from collections import defaultdict
from scipy.stats import ttest_rel

def ensure_list(value):
    """Ensure the value is a list and compute its mean if it's a list of numbers."""
    if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
        return statistics.mean(value)  
    elif isinstance(value, (int, float)):
        return value  
    return None


def calculate_relatedness_difference_and_predict_performance():
    with open('superdata!.json') as f:
        super_data = json.load(f)
    with open('relatedness-data.json') as f:
        relatedness_data = json.load(f)
    
    relatedness_dict = defaultdict(lambda: {'human': {'relatedness': {}}, 'gpt': {'relatedness': {}}})
    board_differences = {}

    # Populate relatedness_dict
    for entry in relatedness_data:
        clue, word, source, relatedness = entry['clue'], entry['word'], entry['source'], entry['relatedness']
        for board, board_entry in super_data.items():
            board_tuple = tuple(board_entry['words'])
            if clue in (board_entry['human_clue'], board_entry['GPT_clue']) and word in board_entry['words']:
                relatedness_dict[board_tuple][source]['relatedness'][word] = relatedness
    # Prepare lists for t-test, ensuring only complete data is used
    relatedness_differences = []
    performance_differences = []

    # Sort boards consistently to ensure correct pairing
    sorted_boards = sorted(super_data.keys())

    for board in sorted_boards:
        board_entry = super_data[board]
        board_tuple = tuple(board_entry['words'])
        human_intended_words = board_entry['human_intended_words']
        gpt_intended_words = board_entry['GPT_intended_words']

        # Ensure that we have relatedness data for all 12 words in the board
        if set(relatedness_dict[board_tuple]['human']['relatedness'].keys()) == set(board_tuple) and \
        set(relatedness_dict[board_tuple]['gpt']['relatedness'].keys()) == set(board_tuple):

            human_means = [
                ensure_list(relatedness_dict[board_tuple]['human']['relatedness'].get(word, None))
                for word in human_intended_words if relatedness_dict[board_tuple]['human']['relatedness'].get(word) is not None
            ]
            gpt_means = [
                ensure_list(relatedness_dict[board_tuple]['gpt']['relatedness'].get(word, None))
                for word in gpt_intended_words if relatedness_dict[board_tuple]['gpt']['relatedness'].get(word) is not None
            ]

            if human_means and gpt_means:
                human_avg = statistics.mean(human_means)
                gpt_avg = statistics.mean(gpt_means)
                relatedness_differences.append(human_avg - gpt_avg)

                human_accuracy = sum([1 for guess in board_entry['human_guess_human_clue'] if guess in human_intended_words])
                gpt_accuracy = sum([1 for guess in board_entry['GPT_guess_GPT_clue'] if guess in gpt_intended_words])
                performance_differences.append(human_accuracy - gpt_accuracy)

    # Debugging: Print the final count of boards included in the t-test
    print(f"\n Number of fully complete boards used in t-test: {len(relatedness_differences)}")

    # Run t-test only on complete data
    if len(relatedness_differences) == len(performance_differences):
        print(f"Paired data verified. Running t-test on {len(relatedness_differences)} samples.")

        # Print each paired sample for verification
        print("\nIndividual Paired Samples (Relatedness Difference → Performance Difference):")
        for i, (r_diff, p_diff) in enumerate(zip(relatedness_differences, performance_differences)):
            print(f"Sample {i+1}: {r_diff:.4f} → {p_diff:.4f}")

        # Run the t-test
        t_stat, p_value = ttest_rel(relatedness_differences, performance_differences)
        print(f"\nT-test results: t-statistic = {t_stat}, p-value = {p_value}")
    else:
        print(f"Error: Mismatch in paired data! Relatedness: {len(relatedness_differences)}, Performance: {len(performance_differences)}")

calculate_relatedness_difference_and_predict_performance()