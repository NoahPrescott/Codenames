# utilizes superdata!.json and relatedness-data.json
# superdata!.json contains each board along with GPT & human clues, GPT & human intended words, and GPT & human guesses
# relatedness-data.json each each judgment comparison as the two words compared, the comparer, and the relatedness score
# relatedness-data.json currently refers to a 'clue' and a 'word'; this is vestigial. These equate to word1 and word2. 

import json
import statistics
from collections import defaultdict
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors


def ensure_list(value):
    """Ensure the value is a list and compute its mean if it's a list of numbers."""
    if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
        return statistics.mean(value)  
    elif isinstance(value, (int, float)):
        return value  
    return None


def have_relatedness_judgments(board_entry, rel_judgments):
    """ Do we have relatedness judgments for this board? """
    words = board_entry['words']
    sources = ['human', 'GPT']
    for source in sources:
        clue = board_entry[source + '_clue']
        if clue not in rel_judgments[source.lower()].keys():
            return False
        for word in words:
            if word not in rel_judgments[source.lower()][clue].keys():
                return False
    return True
        
        
def construct_rel_dict(relatedness_data):
    """ construct dictionary of relatedness judgments from relatedness data"""
    rel_judgments = {'human':{}, 'gpt':{}}
    for d in relatedness_data:
        rel = d['relatedness']
        if rel is None:
            continue
        if isinstance(rel, list):
            rel = sum(rel)/len(rel)
        rel_judgments[d['source']][d['clue']] = rel_judgments[d['source']].get(d['clue'], {})
        rel_judgments[d['source']][d['clue']][d['word']] = rel    
    return rel_judgments


def get_mean_accuracy(intended_words, responses):
    """Compute mean accuracy for given intended words and list of responses"""
    accuracy = []
    # print("intended words, ", intended_words)
    # print("responses, ", responses)
    for res in responses:
        #print("res", res)

        # Ensure res is treated as a list
        if isinstance(res, str):  
            res = [res]  # Convert single word to list

        n_correct = len([w for w in res if w in intended_words])
        # print(n_correct)

        accuracy.append(n_correct / len(res))
    # print(sum(accuracy) / len(accuracy) if accuracy else 0)
    return sum(accuracy) / len(accuracy) if accuracy else 0  # Avoid division by zero


def get_mean_relatedness(clue, intended_words, rel_judgments):
    """ compute mean relatedness between clue and each intended word """
    relatedness = [rel_judgments[clue][w] for w in intended_words]
    return sum(relatedness)/len(relatedness)
    
    
def calculate_relatedness_difference_and_predict_performance():
    with open('../data/raw/gpt/relatedness/combined-data.json') as f:
        super_data = json.load(f)
    with open('../data/raw/gpt/relatedness/relatedness-data.json') as f:
        relatedness_data = json.load(f)
    # construct dictionary with mean relatedness judgments
    rel_judgments = construct_rel_dict(relatedness_data)

    # let's get all boards for which we have relatedness judgment data
    boards = {}
    for board, board_entry in super_data.items():
        if have_relatedness_judgments(board_entry, rel_judgments):
            boards[board] = board_entry


    human_prefers_gpt_boards = []
    gpt_prefers_human_boards = []
    data = {'words':[],  'human_clue':[], 'gpt_clue':[], 'human_human_clue_accuracy':[], 'human_gpt_clue_accuracy':[], 'human_human_clue_relatedness':[], 'human_gpt_clue_relatedness':[], 'gpt_human_clue_accuracy':[], 'gpt_gpt_clue_accuracy':[], 'gpt_human_clue_relatedness':[], 'gpt_gpt_clue_relatedness':[],}
    for board_entry in boards.values():

        # human accuracy on human clues
        data['human_human_clue_accuracy'].append(get_mean_accuracy(board_entry['human_intended_words'], board_entry['human_guess_human_clue']))
        # human accuracy on gpt clues
        data['human_gpt_clue_accuracy'].append(get_mean_accuracy(board_entry['GPT_intended_words'], board_entry['human_guess_GPT_clue']))
        # human relatedness ratings between human clues and intended words
        data['human_human_clue_relatedness'].append(get_mean_relatedness(board_entry['human_clue'], board_entry['human_intended_words'], rel_judgments['human']))
        # human relatedness ratings between gpt clues and intended words
        data['human_gpt_clue_relatedness'].append(get_mean_relatedness(board_entry['GPT_clue'], board_entry['GPT_intended_words'], rel_judgments['human']))

        # gpt accuracy on human clues
        data['gpt_human_clue_accuracy'].append(get_mean_accuracy(board_entry['human_intended_words'], board_entry['gpt_guess_human_clue']))
        # gpt accuracy on gpt clues
        data['gpt_gpt_clue_accuracy'].append(get_mean_accuracy(board_entry['GPT_intended_words'], board_entry['GPT_guess_GPT_clue']))
        # gpt relatedness ratings between human clues and intended words
        data['gpt_human_clue_relatedness'].append(get_mean_relatedness(board_entry['human_clue'], board_entry['human_intended_words'], rel_judgments['gpt']))
        # gpt relatedness ratings between gpt clues and intended words
        data['gpt_gpt_clue_relatedness'].append(get_mean_relatedness(board_entry['GPT_clue'], board_entry['GPT_intended_words'], rel_judgments['gpt']))

        # count up 'off' cases
        if get_mean_accuracy(board_entry['GPT_intended_words'], board_entry['human_guess_GPT_clue']) > get_mean_accuracy(board_entry['human_intended_words'], board_entry['human_guess_human_clue']):
            human_prefers_gpt_boards.append(board_entry)
        if get_mean_accuracy(board_entry['human_intended_words'], board_entry['gpt_guess_human_clue']) > get_mean_accuracy(board_entry['GPT_intended_words'], board_entry['GPT_guess_GPT_clue']):
            gpt_prefers_human_boards.append(board_entry)

    print(f"\n Number of fully complete boards used in t-test: {len(data['human_human_clue_accuracy'])}")
    
    # test for difference in accuracy (human guessers, both giver types)
    print(scipy.stats.ttest_rel(data['human_human_clue_accuracy'], data['human_gpt_clue_accuracy']))
    # plot accuracy
    plt.hist(data['human_human_clue_accuracy'], bins=20, color='blue', alpha=0.6, label='Human accuracy on human clues')
    plt.hist(data['human_gpt_clue_accuracy'], bins=20, color='red', alpha=0.6, label='Human accuracy on GPT clues')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    # test for difference in relatedness (human source)
    print(scipy.stats.ttest_rel(data['human_human_clue_relatedness'], data['human_gpt_clue_relatedness']))
    # plot relatedness
    plt.hist(data['human_human_clue_relatedness'], bins=20, color='blue', alpha=0.6, label='Human relatedness ratings for human clues')
    plt.hist(data['human_gpt_clue_relatedness'], bins=20, color='red', alpha=0.6, label='Human relatedness ratings for GPT clues')
    plt.xlabel('Mean relatedness between clue and intended words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    human_accuracy_diff = [data['human_human_clue_accuracy'][i] - data['human_gpt_clue_accuracy'][i] for i in range(len(data['human_human_clue_accuracy']))]
    human_relatedness_diff = [data['human_human_clue_relatedness'][i] - data['human_gpt_clue_relatedness'][i] for i in range(len(data['human_human_clue_relatedness']))]
    plt.scatter(human_relatedness_diff, human_accuracy_diff, color='blue', marker='o')
    # Labels and title
    plt.xlabel('Human clue relatedness - GPT clue relatedness')
    plt.ylabel('Human clue accuracy - GPT clue accuracy')
    plt.show()
    print(scipy.stats.pearsonr(human_relatedness_diff, human_accuracy_diff))

    gpt_accuracy_diff = [data['gpt_human_clue_accuracy'][i] - data['gpt_gpt_clue_accuracy'][i] for i in range(len(data['gpt_human_clue_accuracy']))]
    gpt_relatedness_diff = [data['gpt_human_clue_relatedness'][i] - data['gpt_gpt_clue_relatedness'][i] for i in range(len(data['gpt_human_clue_relatedness']))]
    plt.scatter(gpt_relatedness_diff, gpt_accuracy_diff, color='blue', marker='o')
    # Labels and title (might be wrong)
    plt.xlabel('Human clue relatedness - GPT clue relatedness')
    plt.ylabel('Human clue accuracy - GPT clue accuracy')
    plt.show()
    print(scipy.stats.pearsonr(gpt_relatedness_diff, gpt_accuracy_diff))
    


    gpt_off_cases = [(i, data['gpt_human_clue_accuracy'][i], data['gpt_gpt_clue_accuracy'][i]) for i in range(len(data['human_gpt_clue_accuracy'])) if data['gpt_human_clue_accuracy'][i] > data['gpt_gpt_clue_accuracy'][i]]
    human_off_cases = [(i, data['human_gpt_clue_accuracy'][i], data['human_human_clue_accuracy'][i]) for i in range(len(data['human_gpt_clue_accuracy'])) if data['human_gpt_clue_accuracy'][i] > data['human_human_clue_accuracy'][i]]

    # redundancy, good calculations match
    print(f"Number of off-cases where GPT does better on human clues: {len(gpt_off_cases)}")
    print(f"Number of off-cases where humans do better on GPT clues: {len(human_off_cases)}")
    print(f"Number of off-cases where GPT does better on human clues: {len(gpt_prefers_human_boards)}")
    print(f"Number of off-cases where humans do better on GPT clues: {len(human_prefers_gpt_boards)}")

    # graphs for off cases
    # identify indices for off-case boards
    gpt_off_case_indices = {i for i, _, _ in gpt_off_cases}
    human_off_case_indices = {i for i, _, _ in human_off_cases}

    # human accuracy differences for off cases
    human_accuracy_diff_off = [
        data['human_human_clue_accuracy'][i] - data['human_gpt_clue_accuracy'][i]
        for i in range(len(data['human_human_clue_accuracy'])) if i in human_off_case_indices
    ]
    human_relatedness_diff_off = [
        data['human_human_clue_relatedness'][i] - data['human_gpt_clue_relatedness'][i]
        for i in range(len(data['human_human_clue_relatedness'])) if i in human_off_case_indices
    ]

    # GPT accuracy differences for off cases
    gpt_accuracy_diff_off = [
        data['gpt_human_clue_accuracy'][i] - data['gpt_gpt_clue_accuracy'][i]
        for i in range(len(data['gpt_human_clue_accuracy'])) if i in gpt_off_case_indices
    ]
    gpt_relatedness_diff_off = [
        data['gpt_human_clue_relatedness'][i] - data['gpt_gpt_clue_relatedness'][i]
        for i in range(len(data['gpt_human_clue_relatedness'])) if i in gpt_off_case_indices
    ]

    # human off-case
    plt.scatter(human_relatedness_diff_off, human_accuracy_diff_off, color='blue', marker='o')

    plt.xlabel('Human clue relatedness - GPT clue relatedness')
    plt.ylabel('Human clue accuracy - GPT clue accuracy')
    plt.title('Scatter Plot of Human Off-Case Differences')
    plt.show()
    print(scipy.stats.pearsonr(human_relatedness_diff_off, human_accuracy_diff_off))


    # GPT off-case
    plt.scatter(gpt_relatedness_diff_off, gpt_accuracy_diff_off, color='red', marker='o')

    plt.xlabel('Human clue relatedness - GPT clue relatednes')  
    plt.ylabel('Human clue accuracy - GPT clue accuracy')  
    plt.title('Scatter Plot of GPT Off-Case Differences')
    plt.show()
    print(scipy.stats.pearsonr(gpt_relatedness_diff_off, gpt_accuracy_diff_off))



calculate_relatedness_difference_and_predict_performance()