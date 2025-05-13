import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
import json
import matplotlib.pyplot as plt


# Given list of relatedness judgments and n,
# compute mean relatedness of top n - mean relatedness of rest
def top_n_vs_rest(rel_list, n):
    rel_list.sort(reverse=True)
    return np.mean(rel_list[:n]) - np.mean(rel_list[n:])

# Given list of relatedness judgments and n,
# compute relatedness of nth most related - relatedness of next most related
def nth_vs_next(rel_list, n):
    rel_list.sort(reverse=True)
    return rel_list[n-1] - rel_list[n]


# Run ttests comparing human and gpt clues based on relatedness judgments 
def compare_human_and_gpt_clues(boards, gpt_clue_gpt_judgment_dict, gpt_clue_human_judgment_dict, human_clue_human_judgment_dict, human_clue_gpt_judgment_dict):

    # For each human/ai pair, store a list of lists: relatedness judgments between clue and words for each word, for each board
    gpt_clue_gpt_judgments = []
    gpt_clue_human_judgments = []
    human_clue_gpt_judgments = []
    human_clue_human_judgments = []

    # human clues
    for idx, board in enumerate(human_clues_boards):
        human_clue = board['clue'].upper()
        words = board['words']
        human_clue_gpt_judgments.append([int(human_clue_gpt_judgment_dict[human_clue][word]) for word in words])
    for human_clue, words in human_clue_human_judgment_dict.items():
        human_clue_human_judgments.append([human_clue_human_judgment_dict[human_clue][word] for word in words])

    # GPT clues
    for idx, board in enumerate(gpt_clues_boards):
        gpt_clue = board['clue'].upper()
        words = board['words']
        gpt_clue_gpt_judgments.append([gpt_clue_gpt_judgment_dict[gpt_clue][word] for word in words])
    for idx, board in enumerate(boards):
        gpt_clue = board['clue'].upper()
        words = board['words']
        gpt_clue_human_judgments.append([np.mean(gpt_clue_human_judgment_dict[gpt_clue][word]) for word in words])

    print(human_clue_human_judgments)
    print(gpt_clue_human_judgments)
    # Compute values of interest for these different lists of relatedness judgments
    gpt_clue_gpt_judgments_numeric = [[int(value) for value in rel_list] for rel_list in gpt_clue_gpt_judgments]

    gpt_gpt_top_3_vs_rest = [top_n_vs_rest(rel_list, n=3) for rel_list in gpt_clue_gpt_judgments_numeric]
    gpt_human_top_3_vs_rest = [top_n_vs_rest(rel_list, n=3) for rel_list in gpt_clue_human_judgments]
    human_gpt_top_3_vs_rest = [top_n_vs_rest(rel_list, n=3) for rel_list in human_clue_gpt_judgments]
    human_human_top_3_vs_rest = [top_n_vs_rest(rel_list, n=3) for rel_list in human_clue_human_judgments]

    gpt_gpt_3_vs_4 = [nth_vs_next(rel_list, n=3) for rel_list in gpt_clue_gpt_judgments_numeric]
    gpt_human_3_vs_4 = [nth_vs_next(rel_list, n=3) for rel_list in gpt_clue_human_judgments]
    human_gpt_3_vs_4 = [nth_vs_next(rel_list, n=3) for rel_list in human_clue_gpt_judgments]
    human_human_3_vs_4 = [nth_vs_next(rel_list, n=3) for rel_list in human_clue_human_judgments]

    # Run t-tests for the first comparison
    print("Human vs. GPT clues, human relatedness judgments:")
    print("Top 3 vs. rest:")
    print(ttest_ind(human_human_top_3_vs_rest, gpt_human_top_3_vs_rest))
    print("3rd vs. 4th:")
    print(ttest_ind(human_human_3_vs_4, gpt_human_3_vs_4))
    min_length = min(len(human_human_3_vs_4), len(gpt_human_3_vs_4))
    array1_truncated = human_human_3_vs_4[:min_length]
    array2_truncated = gpt_human_3_vs_4[:min_length]
    print(ttest_rel(array1_truncated, array2_truncated))  # Paired t-test
    print("")
    print("Teshes")
    print("Length of human_gpt_top_3_vs_rest:", len(human_human_top_3_vs_rest))
    print("Length of gpt_gpt_top_3_vs_rest:", len(gpt_human_top_3_vs_rest))
        # Check lengths before paired t-test
    print("Paired t-test for Human vs. GPT clues, corresponding GPT judgments:")
    print("Length of human_gpt_top_3_vs_rest:", len(human_gpt_top_3_vs_rest))
    print("1Length of gpt_gpt_top_3_vs_rest:", len(gpt_gpt_top_3_vs_rest))
    
    print(ttest_rel(human_gpt_top_3_vs_rest, gpt_gpt_top_3_vs_rest))  # Paired t-test

    print("")

        # Similar check for the 3rd vs 4th comparison
    print("Length of human_gpt_3_vs_4:", len(human_gpt_3_vs_4))
    print("Length of gpt_gpt_3_vs_4:", len(gpt_gpt_3_vs_4))
        
    print(ttest_rel(human_gpt_3_vs_4, gpt_gpt_3_vs_4))
    print("")

    # Create a common bin range for each graph
    def get_common_bin_range(*data_lists):
        min_val = min(min(data) for data in data_lists)
        max_val = max(max(data) for data in data_lists)
        return np.linspace(min_val, max_val, 20)  # 20 bins for each graph

    # Plot the histograms using the same bin edges
    common_bin_range_1 = get_common_bin_range(human_human_3_vs_4, gpt_human_3_vs_4)
    plt.hist(human_human_3_vs_4, bins=common_bin_range_1, alpha=0.5, label="Human Clue, Human Judgment (3rd vs 4th)")
    plt.hist(gpt_human_3_vs_4, bins=common_bin_range_1, alpha=0.5, label="GPT Clue, Human Judgment (3rd vs 4th)")
    plt.legend(loc='upper right')
    plt.xlabel('Difference between 3rd and 4th Relatedness')
    plt.ylabel('Frequency')
    plt.title('Human vs GPT Clue Judgments (3rd vs 4th)')
    plt.show()

    common_bin_range_2 = get_common_bin_range(human_gpt_3_vs_4, gpt_gpt_3_vs_4)
    plt.hist(human_gpt_3_vs_4, bins=common_bin_range_2, alpha=0.5, label="Human Clue, GPT Judgment (3rd vs 4th)")
    plt.hist(gpt_gpt_3_vs_4, bins=common_bin_range_2, alpha=0.5, label="GPT Clue, GPT Judgment (3rd vs 4th)")
    plt.legend(loc='upper right')
    plt.xlabel('Difference between 3rd and 4th Relatedness')
    plt.ylabel('Frequency')
    plt.title('GPT vs GPT Relatedness Judgments (3rd vs 4th)')
    plt.show()

    common_bin_range_3 = get_common_bin_range(human_gpt_top_3_vs_rest, gpt_gpt_top_3_vs_rest)
    plt.hist(human_gpt_top_3_vs_rest, bins=common_bin_range_3, alpha=0.5, label="Human Clue, GPT Judgment (1-3 vs rest)")
    plt.hist(gpt_gpt_top_3_vs_rest, bins=common_bin_range_3, alpha=0.5, label="GPT Clue, GPT Judgment (1-3 vs rest)")
    plt.legend(loc='upper right')
    plt.xlabel('Difference between top and rest')
    plt.ylabel('Frequency')
    plt.title('GPT vs GPT Relatedness Judgments (Top 3 vs Rest)')
    plt.show()

if __name__=="__main__":
    # Load data
    with open('gpt-clue-gpt-relatedness-oct20-averaged-judgments.json') as f:
        gpt_clue_gpt_judgment_dict = json.load(f)
    with open('data/human-data/pair-similarity-gpt-clue/data.json') as f:
        gpt_clue_human_judgment_dict = json.load(f)
    # this is the 80 boards with gpt clues for which we got human and gpt relatedness judgments
    with open('board_subset.json') as f:
        boards = json.load(f)
        
    with open('human-clue-human-judgement-postexp-nov24.json') as f:
        human_clue_human_judgment_dict = json.load(f)
    with open('human-clue-gpt-relatedness-oct10-averaged-judgments.json') as f:
        human_clue_gpt_judgment_dict = json.load(f)

    with open('boards_and_responses.json') as f:
        human_clues_boards = json.load(f)
    with open('gpt4-clues-final-p2-1.0.json') as f:
        gpt_clues_boards = json.load(f)


    # old judgments: 
    # with open('data/gpt-data/pair-similarity-gpt-clue/data.json') as f:
    #     older_gpt_clue_gpt_judgment_dict = json.load(f)
    # with open('data/gpt-data/pair-similarity-human-clue/data.json') as f:
    #     older_human_clue_gpt_judgment_dict = json.load(f)
        
    compare_human_and_gpt_clues(boards, gpt_clue_gpt_judgment_dict, gpt_clue_human_judgment_dict, human_clue_human_judgment_dict, human_clue_gpt_judgment_dict)
    








# other old code below, semi-irrelevant

# import json
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy
# import networkx as nx
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize
# import os
# from scipy.stats import ttest_rel, ttest_ind

# def reformat():
#     with open('data/gpt-data/guess-gpt-clue/gpt4-own-guesses-final-p2-1.0.json') as f:
#         data = json.load(f)
#     for (i, board) in enumerate(data["boards"]):
#         board['gpt-response'] = [r.upper() for r in data['responses'][i].split(", ")]
#     with open('data/gpt-data/guess-gpt-clue/data.json', 'w') as f:
#         json.dump(data["boards"], f)
#     with open('data/gpt-data/guess-human-clue/gpt4-responses.json') as f:
#         data = json.load(f)
#     for (i, board) in enumerate(data["boards"]):
#         board['gpt-response'] = [r.upper() for r in data['responses'][i].split(", ")]
#     with open('data/gpt-data/guess-human-clue/data.json', 'w') as f:
#         json.dump(data["boards"], f)


# def jitter(l):
#     return [v+np.random.uniform(-0.05,0.05) for v in l]


# def get_board_subset():
#     with open('data/gpt-data/guess-gpt-clue/data.json') as f:
#         all_boards = json.load(f)
#     with open('word_pairs.json') as f:
#         word_pairs = json.load(f)
#     board_subset = []
#     for pair_list in word_pairs["wordPairs"]:
#         clue = pair_list[0][0]
#         for p in pair_list:
#             if p[0] != clue:
#                 print("Assertion failed for pair:", p)
#                 print("Expected clue:", clue)
#                 print("In pair list:", pair_list)
#             assert(p[0]==clue)
#         words = [p[1] for p in pair_list]
#         #get board
#         pair_list_boards = []
#         for board in all_boards:
#             if clue == board['clue'].lower():
#                 if len([word for word in board['words'] if word.lower() in words]) == len(board['words']):
#                     pair_list_boards.append(board)
#         assert(len(pair_list_boards)==1)
#         board = pair_list_boards[0]
#         board_subset.append(board)
#     with open('board_subset.json', 'w') as f:
#         json.dump(board_subset, f)


# def get_human_rel_judgments(): 
#     with open('data/human-data/pair-similarity-gpt-clue/experiment_data_1.json') as f:
#         data = json.load(f)
#     data = data["__collections__"]["exptData"]
#     rel_dict = {}
#     for (k, v) in data.items():
#         if "trials" not in v.keys():
#             continue
#         trials = v["trials"][4:84]
#         subj_id = trials[0]["subject_id"]
#         print(subj_id)
#         for trial in trials:
#             w1 = trial["word1"].upper()
#             w2 = trial["word2"].upper()
#             rel_dict[w1] = rel_dict.get(w1, {})
#             rel_dict[w1][w2] = rel_dict[w1].get(w2, []) + [trial["response"]]
#     with open('data/human-data/pair-similarity-gpt-clue/data.json', 'w') as f:
#         json.dump(rel_dict, f)


# def get_guess_from_relatedness_judgments(rel_judgments):
#     top = sorted(rel_judgments.items(), reverse=True, key=lambda x: x[1])
#     words = []
#     ties = []
#     for word_tup in top:
#         best = [t for t in top if t[0] not in words and t[1]>=word_tup[1]]
#         if len(best) > 3 - len(words):
#             ties = [w[0] for w in best]
#             break
#         else:
#             for word in best:
#                 words.append(word[0])
#         if len(words)==3:
#             break
#     weights = [1 for w in words] + [1/len(ties) for w in ties]
#     words = words + ties
#     return words, weights

# def construct_rel_judgment_df():
#     with open('board_subset.json') as f:
#         boards = json.load(f)
#     with open('data/gpt-data/pair-similarity-gpt-clue/data.json') as f:
#         rel_dict = json.load(f)
#     for d in rel_dict.values():
#         for (k, v) in d.items():
#             d[k] = float(v)
#     with open('data/human-data/pair-similarity-gpt-clue/data.json') as f:
#         human_rel_dict = json.load(f)
#     with open('data/human-data/guess-gpt-clue/cluesAndBoardsHumanDataFullTrial.json') as f:
#         human_guesses = json.load(f)
    
#     guess_correct = []
#     rel_correct = []
#     common = []
#     human_guess_correct = []
#     human_rel_correct = []
#     human_common = []
#     for board in boards:
#         # gpt accuracy
#         guess_res = board['gpt-response']
#         guess_correct.append(len([w for w in guess_res if w in board['intended_words']])/3)
        
#         # gpt relatedness
#         if board['clue'] in rel_dict:
#             rel_judgments = {w: rel_dict[board['clue']].get(w, 0) for w in board['words']}
#             words, weights = get_guess_from_relatedness_judgments(rel_judgments)
#             rel_correct.append(np.mean([float(words[i] in board['intended_words'])*weights[i] for i in range(len(words))]))
#             common.append(np.mean([float(words[i] in guess_res)*weights[i] for i in range(len(words))]))

#         # human accuracy
#         k = str(tuple(board['words']+[board['clue']]))
#         human_responses = human_guesses[k]['responses']
#         assert(len([w for w in human_guesses[k]['intended_words'] if w in board['intended_words']])==3)
#         human_accuracy = np.mean([len([w for w in res if w in board['intended_words']]) for res in human_responses])/3
#         human_guess_correct.append(human_accuracy)
        
#         human rel accuracy, human common
#         if board['clue'] in human_rel_dict:
#             rel_judgments = {w: np.mean(human_rel_dict[board['clue']].get(w, [0])) for w in board['words']}
#             words, weights = get_guess_from_relatedness_judgments(rel_judgments)
#             human_rel_correct.append(np.mean([float(words[i] in board['intended_words'])*weights[i] for i in range(len(words))]))
#             common_list = [np.mean([float(words[i] in g)*weights[i] for i in range(len(words))]) for g in human_guesses[k]['responses']]
#             human_common.append(np.mean(common_list))
    
#     df = pd.DataFrame({'gpt_correct': guess_correct, 'human_correct': human_guess_correct, 'gpt_rel_correct': rel_correct, 'human_rel_correct': human_rel_correct, 'gpt_common': common, 'human_common': human_common})
    
#     # Calculate the difference between GPT and human correctness
#     df['correct_diff'] = df['gpt_correct'] - df['human_correct']
    
#     # Rank the boards based on the difference
#     df['gpt_rank'] = df['correct_diff'].rank(ascending=False).astype(int)
    
#     # Tag whether GPT or humans are performing better
#     df['better'] = np.where(df['correct_diff'] > 0, 'GPT', 'Human')
    
#     print("DataFrame:\n", df)  # Debug: Print the entire DataFrame
#     print("Value counts of GPT ranks:\n", df['gpt_rank'].value_counts())  # Debug: Check the distribution of ranks
    
#     return df

# def plot_rel_judgments(df):
#     print(np.mean(df["human_common"]))
#     print(np.mean(df["gpt_common"]))
#     print(scipy.stats.pearsonr(df["human_correct"], df["human_common"]))
#     print(scipy.stats.pearsonr(df["gpt_correct"], df["gpt_common"]))
    
#     # Plot human vs gpt accuracy, colored by gpt rel pred

#     sns.scatterplot(data=df, x="gpt_correct", y="human_correct", hue="gpt_common")#, color=c)
#     #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     #add linear regression line to scatterplot 
#     plt.xlabel("GPT guess accuracy")
#     plt.ylabel("Human guess accuracy")
#     plt.legend(title="Predictiveness of GPT relatedness\njudgments for GPT guesses")
#     #plt.plot(rel, m*rel+b, color="black")
#     plt.show()
    
    
#     # Plot dif bt gpt and human accuracy vs gpt rel pred
#     df["dif"] = df['gpt_correct'] - df['human_correct']
#     m, b = np.polyfit(df["dif"], df["gpt_common"], 1)
#     rel = np.array(df["dif"])
#     print(scipy.stats.pearsonr(df["dif"], df["gpt_common"]))
#     sns.scatterplot(data=df, x="dif", y="gpt_common")#, color=c)
#     #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     #add linear regression line to scatterplot 
#     plt.xlabel("GPT - Human accuracy")
#     plt.ylabel("Predictiveness of GPT relatedness judgments")
#     #plt.legend(title="Predictiveness of GPT relatedness\njudgments for GPT guesses")
#     plt.plot(rel, m*rel+b, color="black")
#     plt.show()

#     # Plot dif bt gpt and human accuracy vs human rel pred
#     df["dif"] = df['gpt_correct'] - df['human_correct']
#     m, b = np.polyfit(df["dif"], df["human_common"], 1)
#     rel = np.array(df["dif"])
#     print(scipy.stats.pearsonr(df["dif"], df["human_common"]))
#     sns.scatterplot(data=df, x="dif", y="human_common")#, color=c)
#     #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     #add linear regression line to scatterplot 
#     plt.xlabel("GPT - Human accuracy")
#     plt.ylabel("Human predictiveness of relatedness judgments")
#     #plt.legend(title="Predictiveness of GPT relatedness\njudgments for GPT guesses")
#     plt.plot(rel, m*rel+b, color="black")
#     plt.show()

# def create_semantic_graphs_for_boards(df):
#     with open('board_subset.json') as f:
#         boards = json.load(f)
#     with open('data/gpt-data/pair-similarity-gpt-clue/data.json') as f:
#         gpt_rel_dict = json.load(f)
#     with open('data/human-data/pair-similarity-gpt-clue/data.json') as f:
#         human_rel_dict = json.load(f)
    
#     graphs = []
    
#     for idx, board in enumerate(boards):
#         clue = board['clue'].upper()
#         words = board['words']
        
#         gpt_graph = nx.Graph()
#         human_graph = nx.Graph()
        
#         for word in words:
#             gpt_graph.add_node(word)
#             human_graph.add_node(word)

#         # for i in range(len(words)):
#             word1 = clue
#             #maybe use the clue var
#             for j in range(len(words)):
#                 word2 = words[j]

#                 # Calculate GPT relatedness weight
#                 if clue in gpt_rel_dict:
#                     gpt_weight = float(gpt_rel_dict[clue][word])
#                     if gpt_weight > 0:
#                         gpt_graph.add_edge(word1, word2, weight=gpt_weight)
#                         print(f"Added edge {word1}-{word2} with weight {gpt_weight:.2f} in gpt graph.")

#                 # Calculate Human relatedness weight
#                 if clue in human_rel_dict:
#                     # Get the relatedness scores for word1
#                     human_rel_word = human_rel_dict[clue].get(word2, [])
                    
#                     # Calculate average relatedness weight
#                     if human_rel_word:
#                         human_weight = np.mean(human_rel_word)
#                     else:
#                         human_weight = 0
                    
#                     if human_weight > 0:
#                         human_graph.add_edge(word1, word2, weight=human_weight)
#                         print(f"Added edge {word1}-{word2} with weight {human_weight:.2f} in human graph.")
        
#         graphs.append((gpt_graph, human_graph, board['clue']))
    
#     return graphs

# def plot_combined_graphs(gpt_graph, human_graph, title, filename, vmin=0, vmax=100):
#     # Create a layout for the two graphs side-by-side
#     fig, axes = plt.subplots(1, 2, figsize=(24, 12))

#     norm = Normalize(vmin=vmin, vmax=vmax)
#     cmap = plt.get_cmap('plasma')

#     # Plot GPT Graph
#     pos = nx.spring_layout(gpt_graph, weight='weight', iterations=1000)
#     weights = nx.get_edge_attributes(gpt_graph, 'weight').values()
#     nx.draw(gpt_graph, pos, with_labels=True, node_color='lightblue', node_size=200,
#             font_size=12, font_weight='bold',
#             edge_color=[cmap(norm(w)) for w in weights],
#             edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax,
#             width=[2 + 8 * norm(w) for w in weights],
#             alpha=0.8, ax=axes[0])
#     axes[0].set_title("GPT Semantic Graph")

#     # Plot Human Graph
#     pos = nx.spring_layout(human_graph, weight='weight', iterations=1000)
#     weights = nx.get_edge_attributes(human_graph, 'weight').values()
#     nx.draw(human_graph, pos, with_labels=True, node_color='lightblue', node_size=200,
#             font_size=12, font_weight='bold',
#             edge_color=[cmap(norm(w)) for w in weights],
#             edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax,
#             width=[2 + 8 * norm(w) for w in weights],
#             alpha=0.8, ax=axes[1])
#     axes[1].set_title("Human Semantic Graph")

#     # Add a single color bar for both graphs
#     sm = ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)

#     # Set the overall title
#     fig.suptitle(title, fontsize=16)
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)  # Adjust to leave space for the main title
#     plt.savefig(filename, format='png', bbox_inches='tight')
#     plt.close(fig)

# # difference graphs
# def collect_differences(boards, gpt_rel_dict, human_rel_dict):
#     comp_differences = {}
#     differences = []
#     for idx, board in enumerate(boards):
#         clue = board['clue'].upper()
#         words = board['words']
        
#         for word in words:
#             if clue in gpt_rel_dict and word in gpt_rel_dict[clue]:
#                 gpt_weight = float(gpt_rel_dict[clue][word])

            
#             if clue in human_rel_dict and word in human_rel_dict[clue]:
#                 human_rel_word = human_rel_dict[clue][word]
#                 if human_rel_word:
#                     human_weight = np.mean(human_rel_word)
            
#             difference_weight = human_weight - gpt_weight
#             differences.append(difference_weight)
    
#     for idx, board in enumerate(boards):
#         clue = board['clue'].upper()
#         words = board['words']
        
#         gpt_scores = []
#         human_scores = []
        
#         # Collect GPT and human relatedness scores
#         if clue in gpt_rel_dict:
#             gpt_scores = {word: float(gpt_rel_dict[clue].get(word, 0)) for word in words}
#         if clue in human_rel_dict:
#             human_scores = {word: np.mean(human_rel_dict[clue].get(word, [0])) for word in words}
        
#         # Sort by relatedness scores
#         gpt_sorted = sorted(gpt_scores.items(), key=lambda x: x[1], reverse=True)
#         human_sorted = sorted(human_scores.items(), key=lambda x: x[1], reverse=True)
        
#         # Ensure board index is added only once
#         if idx not in comp_differences:
#             if len(gpt_sorted) >= 12 and len(human_sorted) >= 12:
#                 # Top 3 and bottom 9 for GPT
#                 top_3_gpt = [score for word, score in gpt_sorted[:3]]
#                 bottom_9_gpt = [score for word, score in gpt_sorted[-9:]]
#                 diff_top_bottom_gpt = np.mean(top_3_gpt) - np.mean(bottom_9_gpt)
                
#                 # Top 3 and bottom 9 for Human
#                 top_3_human = [score for word, score in human_sorted[:3]]
#                 bottom_9_human = [score for word, score in human_sorted[-9:]]
#                 diff_top_bottom_human = np.mean(top_3_human) - np.mean(bottom_9_human)
                
#                 # Differences between top 3 and bottom 9 for GPT and Human
#                 comp_differences[idx] = {
#                     'diff_top_bottom_gpt': diff_top_bottom_gpt,
#                     'diff_top_bottom_human': diff_top_bottom_human
#                 }
            
#             # Comparison between 3rd and 4th related words
#             if len(gpt_sorted) >= 4 and len(human_sorted) >= 4:
#                 third_gpt = gpt_sorted[2][1]
#                 fourth_gpt = gpt_sorted[3][1]
#                 third_human = human_sorted[2][1]
#                 fourth_human = human_sorted[3][1]
#                 comp_differences[idx].update({
#                     'diff_third_fourth_gpt': third_gpt - fourth_gpt,
#                     'diff_third_fourth_human': third_human - fourth_human
#                 })
    
#     mean_diff = np.mean(differences)
#     std_diff = np.std(differences)
    
#     return differences, comp_differences, mean_diff, std_diff

# def create_combined_diff_graphs(df, mean_diff, std_diff):
#     with open('board_subset.json') as f:
#         boards = json.load(f)
#     with open('data/gpt-data/pair-similarity-gpt-clue/data.json') as f:
#         gpt_rel_dict = json.load(f)
#     with open('data/human-data/pair-similarity-gpt-clue/data.json') as f:
#         human_rel_dict = json.load(f)

#     target_clue = 'PROTECTION'  # Define the clue you want to debug
#     graphs = []
    
#     for idx, board in enumerate(boards):
#         clue = board['clue'].upper()
#         words = board['words']
        
#         difference_graph = nx.Graph()
        
#         # Debug: Print responses for the target clue
#         if clue == target_clue:
#             print(f"Debugging clue: {clue}")
#             print("GPT responses:")
#             if clue in gpt_rel_dict:
#                 for word in words:
#                     gpt_weight = float(gpt_rel_dict[clue].get(word, 0))
#                     print(f"Word: {word}, GPT Weight: {gpt_weight:.2f}")

#             print("Human responses:")
#             if clue in human_rel_dict:
#                 for word in words:
#                     human_weights = human_rel_dict[clue].get(word, [])
#                     avg_human_weight = np.mean(human_weights) if human_weights else 0
#                     print(f"Word: {word}, Human Weight: {avg_human_weight:.2f}")
#             else:
#                 print("No Human responses found for this clue.")

#         for word in words:
#             difference_graph.add_node(word)

#         word1 = clue
        
#         for word2 in words:
#             if clue in gpt_rel_dict and word2 in gpt_rel_dict[clue]:
#                 gpt_weight = float(gpt_rel_dict[clue][word2])
#             else:
#                 gpt_weight = 0

#             if clue in human_rel_dict and word2 in human_rel_dict[clue]:
#                 human_rel_word = human_rel_dict[clue][word2]
#                 if human_rel_word:
#                     human_weight = np.mean(human_rel_word)
#                 else:
#                     human_weight = 0
#             else:
#                 human_weight = 0

#             difference_weight = human_weight - gpt_weight
#             z_score = (difference_weight - mean_diff) / std_diff

#             # Debug: Print the calculated values
#             print(f"Word: {word2}, GPT Weight: {gpt_weight:.2f}, Human Weight: {human_weight:.2f}, Difference Weight: {difference_weight:.2f}, Z-score: {z_score:.2f}")
            
#             if z_score > 0:
#                 color = 'green'
#             elif z_score < 0:
#                 color = 'red'
#             else:
#                 color = 'green'
#                 z_score = 0.1  # otherwise the graph looks bad

#             difference_graph.add_edge(word1, word2, weight=abs(z_score), color=color)
        
#         graphs.append((difference_graph, board['clue']))
    
#     return graphs


# def plot_combined_diff_graph(graph, title, filename, gpt_rank, hum_cor, gpt_cor, diff_top_bottom_gpt, diff_top_bottom_human, diff_third_fourth_gpt, diff_third_fourth_human):
#     try:
#         pos = nx.spring_layout(graph, weight='weight', iterations=1000, k=0.5, scale=-1.0)

#         max_weight = max([graph[u][v]['weight'] for u, v in graph.edges()])
#         norm = Normalize(vmin=0, vmax=max_weight)

#         cmap_red = plt.get_cmap('Reds')
#         cmap_green = plt.get_cmap('Greens')

#         edges_red = [(u, v) for u, v, d in graph.edges(data=True) if d['color'] == 'red']
#         edges_green = [(u, v) for u, v, d in graph.edges(data=True) if d['color'] == 'green']

#         edge_weights_red = [graph[u][v]['weight'] for u, v in edges_red]
#         edge_weights_green = [graph[u][v]['weight'] for u, v in edges_green]

#         fig, ax = plt.subplots(figsize=(8, 8))

#         nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=200,
#                 font_size=12, font_weight='bold', edge_color='black', width=1, alpha=0.8)

#         nx.draw_networkx_edges(graph, pos, edgelist=edges_red, width=2,
#                                edge_color=[cmap_red(norm(w)) for w in edge_weights_red], alpha=0.8)

#         nx.draw_networkx_edges(graph, pos, edgelist=edges_green, width=2,
#                                edge_color=[cmap_green(norm(w)) for w in edge_weights_green], alpha=0.8)

#         sm_red = ScalarMappable(cmap=cmap_red, norm=norm)
#         sm_red.set_array([])
#         fig.colorbar(sm_red, ax=ax, orientation='horizontal', fraction=0.043, pad=0.1, label='Weight Difference (Red: GPT preference over human)')

#         sm_green = ScalarMappable(cmap=cmap_green, norm=norm)
#         sm_green.set_array([])
#         fig.colorbar(sm_green, ax=ax, orientation='horizontal', fraction=0.05, pad=0.1, label='Weight Difference (Green: Human preference over GPT)')

#         # Add GPT score text and calculated differences
#         total_boards = len(df)
#         gpt_score_text = f"GPT score: {gpt_rank}/{total_boards}"
#         gpt_cor_text = f"GPT correct: {gpt_cor}"
#         human_cor_text = f"Human correct: {hum_cor}"
        
#         if diff_top_bottom_gpt is not None:
#             diff_text_top_3_vs_bottom_9_gpt = f"Top 3 vs Bottom 9 Difference GPT: {diff_top_bottom_gpt:.2f}"

#         if diff_top_bottom_human is not None:
#             diff_text_top_3_vs_bottom_9_human = f"Top 3 vs Bottom 9 Difference Human: {diff_top_bottom_human:.2f}"

#         if diff_third_fourth_gpt is not None:
#             diff_text_third_vs_fourth_gpt = f"3rd vs 4th Difference GPT: {diff_third_fourth_gpt:.2f}"

#         if diff_third_fourth_human is not None:
#             diff_text_third_vs_fourth_human = f"3rd vs 4th Difference Human: {diff_third_fourth_human:.2f}"

#         plt.title(f"{title}\n{gpt_score_text}\n{gpt_cor_text}\n{human_cor_text}\n{diff_text_top_3_vs_bottom_9_gpt}\n{diff_text_top_3_vs_bottom_9_human}\n{diff_text_third_vs_fourth_gpt}\n{diff_text_third_vs_fourth_human}")

#         plt.tight_layout()
#         plt.subplots_adjust(top=0.9)

#         os.makedirs(os.path.dirname(filename), exist_ok=True)

#         plt.savefig(filename, format='png', bbox_inches='tight')
#         plt.close(fig)

#     except Exception as e:
#         print(f"Failed to save graph for {title} to {filename}")
#         print(f"Error: {e}")

# def perform_t_tests(comp_differences):
#     t_tests_results = {
#         '3v4': {'t_statistic': None, 'p_value': None},
#         '1-3vRest': {'t_statistic': None, 'p_value': None}
#     }

#     third_vs_fourth_gpt = []
#     third_vs_fourth_human = []
#     top_3_vs_rest_gpt = []
#     top_3_vs_rest_human = []

#     for idx, diffs in comp_differences.items():
#         if 'diff_third_fourth_gpt' in diffs and 'diff_third_fourth_human' in diffs:
#             third_vs_fourth_gpt.append(diffs['diff_third_fourth_gpt'])
#             third_vs_fourth_human.append(diffs['diff_third_fourth_human'])
        
#         if 'diff_top_bottom_gpt' in diffs and 'diff_top_bottom_human' in diffs:
#             top_3_vs_rest_gpt.append(diffs['diff_top_bottom_gpt'])
#             top_3_vs_rest_human.append(diffs['diff_top_bottom_human'])

#     # t-test for 3rd vs 4th related words
#     t_stat_3v4, p_value_3v4 = ttest_rel(third_vs_fourth_gpt, third_vs_fourth_human)
#     t_tests_results['3v4']['t_statistic'] = t_stat_3v4
#     t_tests_results['3v4']['p_value'] = p_value_3v4

#     # t-test for top 3 vs rest (1-3 v Rest)
#     t_stat_1_3vRest, p_value_1_3vRest = ttest_rel(top_3_vs_rest_gpt, top_3_vs_rest_human)
#     t_tests_results['1-3vRest']['t_statistic'] = t_stat_1_3vRest
#     t_tests_results['1-3vRest']['p_value'] = p_value_1_3vRest
#     # Plotting histograms
#     plt.figure(figsize=(12, 6))

#     # Histogram for 3v4 values
#     plt.subplot(1, 2, 1)
#     plt.hist(human_3v4_values, bins=10, alpha=0.7, label='Human 3v4', color='blue')
#     plt.hist(gpt_3v4_values, bins=10, alpha=0.7, label='GPT 3v4', color='orange')
#     plt.title('Histogram of 3v4 Values')
#     plt.xlabel('3v4 Difference')
#     plt.ylabel('Frequency')
#     plt.legend()

#     # Histogram for 1-3vRest values
#     plt.subplot(1, 2, 2)
#     plt.hist(human_1_3vRest_values, bins=10, alpha=0.7, label='Human 1-3vRest', color='blue')
#     plt.hist(gpt_1_3vRest_values, bins=10, alpha=0.7, label='GPT 1-3vRest', color='orange')
#     plt.title('Histogram of 1-3vRest Values')
#     plt.xlabel('1-3vRest Difference')
#     plt.ylabel('Frequency')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()
#     return t_tests_results

# if __name__ == "__main__":
#     df = construct_rel_judgment_df()
    
#     with open('board_subset.json') as f:
#         boards = json.load(f)
#     with open('data/gpt-data/pair-similarity-gpt-clue/data.json') as f:
#         gpt_rel_dict = json.load(f)
#     with open('data/human-data/pair-similarity-gpt-clue/data.json') as f:
#         human_rel_dict = json.load(f)

#     # Calculate relatedness differences
#     differences, comp_differences, mean_diff, std_diff = collect_differences(boards, gpt_rel_dict, human_rel_dict)

#     # Create combined difference graphs
#     os.makedirs('Combined_Difference_Graphs_Normalized', exist_ok=True)
#     clue_count = {}

#     graphs = create_combined_diff_graphs(df, mean_diff, std_diff)
#     for idx, (difference_graph, clue) in enumerate(graphs):
#         sanitized_clue = clue.replace('/', '-')
#         clue_count[sanitized_clue] = clue_count.get(sanitized_clue, 0) + 1
#         suffix = '' if clue_count[sanitized_clue] == 1 else str(clue_count[sanitized_clue])
        
#         gpt_rank = df.loc[idx, 'gpt_rank']
#         hum_cor = df.loc[idx, 'human_correct']
#         gpt_cor = df.loc[idx, 'gpt_correct']
#         filename = f"Combined_Difference_Graphs_Normalized/{gpt_rank}{sanitized_clue}{suffix}.png"
#         t_tests_results = perform_t_tests(comp_differences)
#         print("T-Tests Results:")
#         print(t_tests_results)        
#         plot_combined_diff_graph(difference_graph, clue, filename,
#                                  df.loc[idx, 'gpt_rank'], 
#                                  df.loc[idx, 'human_correct'], 
#                                  df.loc[idx, 'gpt_correct'],
#                                  comp_differences[idx]['diff_top_bottom_gpt'],
#                                  comp_differences[idx]['diff_top_bottom_human'],
#                                  comp_differences[idx]['diff_third_fourth_gpt'],
#                                  comp_differences[idx]['diff_third_fourth_human'],
#                                  )
#     # os.makedirs('Combined_Semantic_Graphs', exist_ok=True)
    
#     # clue_count = {}
#     # similarity_results = []

#     # graphs = create_semantic_graphs_for_boards(df)
#     # for idx, (gpt_graph, human_graph, clue) in enumerate(graphs):
#     #     sanitized_clue = clue.replace('/', '-')
#     #     clue_count[sanitized_clue] = clue_count.get(sanitized_clue, 0) + 1
#     #     suffix = '' if clue_count[sanitized_clue] == 1 else str(clue_count[sanitized_clue])

#     #     # Check if GPT accuracy is greater than human accuracy
#     #     if df.loc[idx, 'gpt_correct'] > df.loc[idx, 'human_correct']:
#     #         # Print the compared words, GPT correct, and human correct
#     #         compared_words = ", ".join(graphs[idx][1].nodes)  # Print nodes (words) in the human graph
#     #         gpt_correct = df.loc[idx, 'gpt_correct']
#     #         human_correct = df.loc[idx, 'human_correct']
            
#     #         print(f"Compared Words: {compared_words}")
#     #         print(f"GPT Correctness: {gpt_correct}")
#     #         print(f"Human Correctness: {human_correct}")

#     #         combined_filename = f"Combined_Semantic_Graphs/{sanitized_clue}{suffix}.png"
#     #         plot_combined_graphs(gpt_graph, human_graph, f"Semantic Graphs for Clue '{clue}'", combined_filename)
