#!/usr/bin/env python3
import json
import numpy as np

#keep trials for which participants correctly remembered their guess
def getGoodTrials(trials):
    new_trials = []
    for t in trials:
        id = t['subject_id']
        trial_num = int(t['trial_order'])-1
        #see if this subject made it thru to the end to give subject data (throwing out if not rn)
        subj_temp = list(filter(lambda d: d['subject_id']==id, subj_data))
        if(len(subj_temp)>0):
            subj=subj_temp[0]
        else:
            continue
        #check subject data to see if they passed the check for this trial
        #continue if subject failed this trial by getting 2 wrong on check
        if trial_num in subj['check_idxs']:
            check_idx = subj['check_idxs'].index(trial_num)
            if subj['check_num_dif'][check_idx] >= 2:
                continue
        #add trial if subject failed less than 4 trials
        if len(list(filter(lambda n: n >= 2, subj['check_num_dif']))) < 4:
            new_trials.append(t)
    return new_trials

#take a look at data and make sure something hasn't gone horribly wrong
def sanityCheck(trials):
    for t in trials[50:60]:
        print('Clue: ' + t['clue'])
        print(t['words'][0:4])
        print(t['words'][4:8])
        print(t['words'][8:12])
        print('\n\n\n\n\n\n\n\n\n\n')
        print(t['chosen_words'])
        print('\n\n\n\n')


def writeCluesAndBoardsToFile(trials):
    board_responses = {}

    for t in trials:
        board_key = str(tuple(t['words'] + [t['clue']]))  
        intended_words = t['intended_words'] if isinstance(t['intended_words'], list) else eval(t['intended_words'])

        if board_key not in board_responses:
            board_responses[board_key] = {
                'clue': t['clue'].upper(),
                'words': t['words'],
                'intended_words': intended_words,
                'responses': []  # Initialize an empty list for responses
            }
        
        board_responses[board_key]['responses'].append(t['chosen_words'])


    with open('cluesAndBoardsHumanDataFullTrial.json', 'w') as f:
        json.dump(board_responses, f)



#rewrite subject and trial data files with desired data types and return data
def getData():
    with open('human_data/full-run-1/trials.json') as f:
        td = json.load(f)
    with open('human_data/full-run-1/subjects.json') as f:
        sd = json.load(f)
    #fix up trial data
    for i in range(len(td)):  
        td[i]['words'] = eval(td[i]['words'])
        td[i]['chosen_words'] = eval(td[i]['chosen_words'])
        td[i]['rt'] = float(td[i]['rt'])
        td[i]['trial_order'] = int(td[i]['trial_order'])
    #fix up subject data
    for i in range(len(sd)):  
        sd[i]['check_words'] = eval(sd[i]['check_words'])
    return td, sd

def printCorrectRough(trials):
    total_trials = len(trials)
    total_correct = sum(int(t["num_correct"]) for t in trials)
    max_possible_correct = total_trials * 3  # Assuming max correct is 3 for each trial

    correctness_percentage = (total_correct / max_possible_correct) * 100

    print(f"Total Trials: {total_trials}")
    print(f"Total Correct: {total_correct}")
    print(f"Correctness Percentage: {correctness_percentage:.2f}%")

data, subj_data = getData()
tr = getGoodTrials(data)
printCorrectRough(tr)
writeCluesAndBoardsToFile(tr)


# to convert a boards_and_responses.json type file:
# with open('boards_and_responses.json') as m:
#     human_guess_human_clue = json.load(m)
# writeCluesAndBoardsToFile(human_guess_human_clue)

# def writeCluesAndBoardsToFile(trials):
#     board_responses = {}

#     for t in trials:
#         board_key = str(tuple(t['words'] + [t['clue']]))  
#         intended_words = t['intended_words'] if isinstance(t['intended_words'], list) else eval(t['intended_words'])

#         if board_key not in board_responses:
#             board_responses[board_key] = {
#                 'clue': t['clue'].upper(),
#                 'words': t['words'],
#                 'intended_words': intended_words,
#                 'responses': []  # Initialize an empty list for responses
#             }
        
#         board_responses[board_key]['responses'].extend(t['responses'])

#     with open('cluesAndBoardsHumanDataHumanClueReformat.json', 'w') as f:
#         json.dump(board_responses, f)
