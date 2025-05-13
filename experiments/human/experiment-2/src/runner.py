
import re
import json
# IMPORTANT: UPLOAD reformatted_word_pairs.json to RESET THE COUNTS FOR ALL WORD PAIRS. 

# Load the JSON data from the file
with open('reformatted_word_pairs.json', 'r') as file:
    data = json.load(file)

# Process the data to the desired format
reformatted_data = []
for pair in data["__collections__"]["word-pairs"]:
    for key, value in pair.items():
        if value:  # Only add non-empty pairs
            reformatted_data.append(value)

# Structure the output to include __collections__ with the collection named word-pairs
output_data = {
    "__collections__": {
        "word-pairs": reformatted_data
    }
}

# Write the modified data back to the JSON file
with open('reformatted_word_pairs.json', 'w') as file:
    json.dump(output_data, file, indent=2)

# Optionally print the reformatted JSON
print(json.dumps(output_data, indent=2))




# # Let's take word pairs and make them into the necessary word1 word2 count format:
# # Read and preprocess the JavaScript file
# with open('word_pairs.js', 'r') as file:
#     # Read the content and remove the JavaScript-specific part
#     content = file.read()
#     # Remove "export const pair_sets = " and any surrounding whitespace
#     content = re.sub(r'^export const pair_sets = ', '', content).strip()
#     # Remove trailing semicolon if there is one
#     content = content.rstrip(';')

#     # Now parse the JSON-like array content
#     pair_sets = json.loads(content)

# # Convert the pair sets to the desired JSON structure
# word_pairs = []
# for group in pair_sets:
#     for pair in group:
#         word_pairs.append({
#             "word1": pair[0],
#             "word2": pair[1],
#             "count": 0
#         })

# output_data = {
#     "__collections__": {
#         "word-pairs": word_pairs
#     }
# }

# # Write to a JSON file
# with open('reformatted_word_pairs.json', 'w') as file:
#     json.dump(output_data, file, indent=2)

# # Optionally, print the output to verify
# print(json.dumps(output_data, indent=2))
