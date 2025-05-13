import os
import openai
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm

# these are for compared_judgments and might be moved later
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import scipy as scipy
from scipy.stats import pearsonr


# prompts
query_opener = "I am going to give you a one-word clue, along with a list of 12 words. I chose the clue to help you guess exactly 3 of the words in the list. Your task is to list the 3 words that you think I have in mind based on the clue."
query_closer = "Please simply list the 3 words that you think I have in mind based on the clue."

# clue prompt 1
# query_clue_opener = "I am going to provide you with a list of words. Your task is to identify a word that would allow a guesser to identify 3 specific words (of your choice) from the board. Please provide the 3 target words as well as the clue word you have come up with."
# query_clue_closer = "Please give your response in this format: CLUE. WORD,WORD,WORD"

# clue prompt 2
query_clue_opener = "You will be shown a board of 12 words. Your task is to select 3 of the words, and come up with a one-word clue that would allow another player to guess those words. After you choose the 3 words and give your clue, another player will be shown the same board of 12 words and your clue, and will be asked to guess the 3 words that you had in mind. So, your clue should apply to each of the 3 words you chose more than it applies to any of the other words."
query_clue_closer = "Please give your response in this format: CLUE. WORD,WORD,WORD"

# clue prompt 3
# query_clue_opener = "Let's play a game called codenames! In this game, we both see a list of 12 words. You're the spymaster, which means your job is to come up with a one-word clue that points to 3 words in the list. I will try to guess the 3 words based on your clue."
# query_clue_closer = "Please give your response in this format: CLUE. WORD,WORD,WORD"

query_relatedness_opener = "I am going to provide you with 2 words. Your task is to assess them on a relatedness scale of 1 to 100, 100 being very related and 0 being not related."
query_relatedness_closer = "Please just provide your score."

def get_response(query):
    max_tries = 10 #  max tries
    for i in range(max_tries):
        try:
            chat_completion = openai.ChatCompletion.create(
                                            model="gpt-4o",
                                            temperature=0,
                                            messages=[{"role": "user",
                                                    "content": query}])
            return chat_completion.choices[0].message.content
        except openai.error.Timeout as e:
            #retry
            print(f"OpenAI API returned an API Error: {e}")
            print(f"Trying again, attempt #{(i+1)}")  # This should never increment, since it won't timeout twice in a row





def get_query(clue, word_list):
    q = query_opener
    q += "\n"
    q += "Clue: " + clue.lower()
    q += "\n"
    q += "Words: " + ', '.join([w.lower() for w in word_list])
    q += "\n"
    q += query_closer
    print(q)
    return q

def get_gpt_responses_guess(input, output):

    # with open('gpt4-responses.json') as f:  # comment this out (and comment in data = {"boards": [], "responses": []}) if starting new data collection
    #     data = json.load(f)
    data = {"boards": [], "responses": []}

    #get the codenames boards
    with open(input) as f:
        all_boards = json.load(f)

    #all_boards = all_boards[350:400]  # adjusts which boards are queried. Comment out to run all boards.

    #go through all the codenames boards
    for board in all_boards:
        data["boards"].append(board)

        query = get_query(board["clue"], board["words"])
        response = get_response(query)
        print(response)
        data["responses"].append(response)
    
    #store data in json
    with open(output, 'w') as f:
        json.dump(data, f)











# get gpt to give a clue
def clue_query(word_list):
    q = query_clue_opener
    q += "\n"
    q += "Words: " + ', '.join([w.lower() for w in word_list])
    q += "\n"
    q += query_clue_closer
    print(q)
    return q

def get_gpt_responses_clue(file):
    data = {"boards": [], "responses": []} # only works as full batch right now

    with open('boards_and_responses.json') as f:
        all_boards = json.load(f)

        # all_boards = all_boards[399:400]  # adjusts which boards are queried. Comment out to run all boards.

    for board in all_boards:
        data["boards"].append(board)
        query = clue_query(board["words"])
        response = get_response(query)
        print(response)
        data["responses"].append({"board": board["words"], "response": response})

    with open(file, 'w') as f:
        json.dump(data, f)

def parse_response(file):
    with open(file) as f:
        data = json.load(f)
    parsed_responses = []
    for response in data["responses"]:

        [clue, words] = response["response"].split(".") 
        words = words.split(",")
        words = [word.strip().upper() for word in words]
        clue = clue.strip().upper()
        board = response["board"]

        parsed_responses.append({"clue": clue, "words": board, "intended_words": words})
    data = parsed_responses
    
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)








# get gpt to give word relatedness
def relatedness_query(clue, word):
    q = query_relatedness_opener
    q += "\n"
    q += "Words: " + word.lower() + ', ' + clue.lower()
    q += "\n"
    q += query_relatedness_closer
    # print(q)
    return q

def get_word_relatedness_scores(save_file):
    data = {} # only works as full batch right now

    # with open('boards_and_responses.json') as f:
    with open('gpt4-clues-final-p2-1.0.json') as f:
        all_boards = json.load(f)

        #all_boards = all_boards[399:400]  # adjusts which boards are queried. Comment out to run all boards.

    for board in all_boards:
        data[board["clue"]] = data.get(board["clue"], {})
        for word in board["words"]:
            query = relatedness_query(board["clue"], word)
            print(query)
            response = get_response(query)
            print(response)
            data[board["clue"]][word] = response


    with open(save_file, 'w') as f:
        json.dump(data, f)


def compare_relatedness(gpt_related_file, human_related_file):
    data = {}

    with open(gpt_related_file) as gptf:
        gpt_data = json.load(gptf)

    with open(human_related_file) as hf:
        human_data = json.load(hf)

    for board in gpt_data:
        if board in human_data:
            for word in gpt_data[board]:

                if word in human_data[board]:
                    comparison = board + ', ' + word
                    gpt_value = float(gpt_data[board][word])
                    human_value = float(human_data[board][word])
                    difference = gpt_value - human_value

                    data[comparison] = {
                            "GPT": gpt_value,
                            "Human": human_value,
                            "Difference": difference
                    }
    # above saves data in a nested format that I find readable. Now we will convert into tabular format
    reshaped_data = []
    for key, values in data.items():
        comparison_1, comparison_2 = key.split(", ")
        entry = {
            "Comparison_1": comparison_1,
            "Comparison_2": comparison_2,
            "GPT": values["GPT"],
            "Human": values["Human"],
            "Difference": values["Difference"]
        }
        reshaped_data.append(entry)
    df = pd.DataFrame(reshaped_data)
    df.to_csv('compared_judgments.csv')

    # Correlation matrix
    columns_of_interest = ['GPT', 'Human']
    correlation_matrix = df[columns_of_interest].corr()
    # Display the correlation matrix
    print(correlation_matrix)

    # Correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    plt.show()

    # Scatterplot
    # Assuming df is your DataFrame with "Comparison_1," "Comparison_2," "GPT," and "Human" columns
    fig = px.scatter(df, x='GPT', y='Human', hover_data=['Comparison_1', 'Comparison_2'], title='Scatter Plot: GPT vs Human')
    fig.show()

    # Pearson Correlation
    correlation, p_value = pearsonr(df['GPT'], df['Human'])
    # Display the correlation coefficient and p-value
    print(f'Pearson correlation coefficient: {correlation}')
    print(f'P-value: {p_value}')





# get gpt to guess target words from a list of words (and clue)
# get_gpt_responses_guess('gpt4-clues-final-p2-1.0.json', 'gpt4-own-guesses-final-p2-1.0.json')

# get gpt to select 3 words from the board and give a corresponding clue
#get_gpt_responses_clue('gpt4-clues-final-p2-1.0.json')
# parse_response('gpt4-clues-final-p2-1.0.json')

# get gpt word-relatedness scores for each clue-word combo of a board
#get_word_relatedness_scores('human-clue-gpt-relatedness-oct10batch-1.json')
# get_word_relatedness_scores('human-clue-gpt-relatedness-oct10batch-2.json')
# get_word_relatedness_scores('human-clue-gpt-relatedness-oct10batch-3.json')
# get_word_relatedness_scores('human-clue-gpt-relatedness-oct10batch-4.json')
get_word_relatedness_scores('gpt-clue-gpt-relatedness-oct20batch-1.json')
get_word_relatedness_scores('gpt-clue-gpt-relatedness-oct20batch-2.json')
get_word_relatedness_scores('gpt-clue-gpt-relatedness-oct20batch-3.json')
get_word_relatedness_scores('gpt-clue-gpt-relatedness-oct20batch-4.json')

# compare relatedness scores of human and gpt (only accesses the shared scores)
# compare_relatedness('gpt4-relatedness-boards-and-responses.json', 'human-relatedness-judgments.json')