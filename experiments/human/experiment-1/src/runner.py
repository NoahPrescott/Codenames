import json
# Upload reformatted_word_pairs.json to RESET THE COUNTS FOR ALL WORD PAIRS. 
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
