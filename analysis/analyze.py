import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#clue, words, intended_words, responses
#what percent are correct?
#want a df with board_idx, corr, guesser (human or gpt4)
def get_gpt_correct(all, intended, res):
    res = res.split(',')
    res = [r.strip().upper() for r in res]
    for r in res:
        if r not in all:
            print(all)
            print(r)
    assert(len(res)==3)
    return len([r for r in res if r in intended])/3

def make_correct_df():
    with open('./data/raw/gpt/guess-on-gpt-clue/gpt4-guess-to-own-clue.json') as f:
        res_data = json.load(f)
    data = []
    for i, board in enumerate(res_data['boards']):
        gpt_correct = get_gpt_correct(board['words'], board['intended_words'], res_data['responses'][i])
        #data.append({'idx':i, 'correct':gpt_correct, 'guesser':'gpt4'})
        for res in board['responses']:
            human_correct = len([r for r in res if r in board['intended_words']])/3
            data.append({'idx':i, 'human_correct':human_correct, 'gpt4_correct':gpt_correct})
    df = pd.DataFrame.from_dict(data)
    df.to_csv('correct.csv')

def calculate_correct_words(response, intended_words):
    response_words = set(response.upper().split(", "))
    correct_words = len(response_words.intersection(intended_words))
    return correct_words

def make_correct_simple(file):
    with open(file) as f:
        res_data = json.load(f)
    total_words = 0
    correct_words = 0

    for board, response in zip(res_data["boards"], res_data["responses"]):
        intended_words = set(board["intended_words"])
        correct_words += calculate_correct_words(response, intended_words)
        total_words += len(intended_words)

    correctness_percentage = (correct_words / total_words)

    return correctness_percentage


def compare_guesses():
    with open('../data/raw/gpt/guess-on-human-clue/gpt4-responses.json') as f:
        res_data = json.load(f)
    data = []
    for i, board in enumerate(res_data['boards']):
        human_responses = board['responses']
        gpt4_response = res_data['responses'][i]
        gpt4_response = gpt4_response.split(',')
        gpt4_response = [r.strip().upper() for r in gpt4_response]
        for subj_index, response in enumerate(human_responses):
            gpt4_common = get_common(response, gpt4_response)
            for subj2_index, response2 in enumerate(human_responses):
                if subj_index == subj2_index:
                    continue
                subj_common = get_common(response, response2)
                data.append({'idx':i, 'comparison_subject':subj_index,'gpt4-common':gpt4_common,'human-common':subj_common})
    df = pd.DataFrame.from_dict(data)
    df.to_csv('comparison.csv')
    means = df.groupby(['idx']).mean().reset_index()
    print(np.mean(means['gpt4-common']))
    print(np.mean(means['human-common']))

def get_common(l1, l2):
    count = 0
    for r in l1:
        if r in l2:
            count += 1
    return count

def plot_correct():
    df = pd.read_csv('correct.csv')

    means = df.groupby('idx').mean().reset_index()
    print(means.head())
    print(np.mean(means['human_correct']))
    print(np.mean(means['gpt4_correct']))
    print(np.corrcoef(means['human_correct'], means['gpt4_correct']))
    stds = df.groupby('idx').std().reset_index()

    # Create the scatterplot with means and error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=means, x='human_correct', y='gpt4_correct',  ax=ax)
    #for i, (_, row) in enumerate(means.iterrows()):
    #    ax.errorbar(row['human_correct'], row['gpt4_correct'], xerr=stds.loc[i, 'human_correct'], yerr=stds.loc[i, 'gpt4_correct'], fmt='none', color='black')

    # Show the plot
    plt.show()

#make_correct_df()
#plot_correct()  # make_correct must be ran first (correct.csv must be current) for intended effect
#compare_guesses()
# print(make_correct_simple('cluesAndBoardsNewData.json'))
