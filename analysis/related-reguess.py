#!/usr/bin/env python3

import os
import openai
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm

def relatedness_as_guess(relatedness_file):
    data = {"boards": [], "responses": []}

    with open('gpt4-clues.json') as f:
        all_boards = json.load(f)
    # all_boards = all_boards[395:400]  # adjusts which boards are queried. Comment out to run all boards.
    #go through all the codenames boards
    for board in all_boards:
        data["boards"].append(board)

        clue = board["clue"]
        guesses = get_guess(relatedness_file, clue, board)
        data["responses"].append(guesses)

    with open('gpt4-relatedness-guesses-own-clue.json', 'w') as f:
        json.dump(data, f)



# get top 3 guesses
def get_guess(file_name, clue, board):
    with open(file_name, 'r') as f:
        relatedness_data = json.load(f)

    # Extract relatedness scores for the given clue
    if clue in relatedness_data:
        clue_relatedness = relatedness_data[clue]
        top_words = sorted([word for word in board['words']], key=lambda w: int(clue_relatedness.get(w, 0)), reverse=True)[:3]
        print(top_words)
        return ", ".join(top_words)
    else:
        return ""

    
relatedness_as_guess('gpt4-responses.json')