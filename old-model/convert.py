import json

input_file = "2.jsonl"

output_file = "2_chat.jsonl"

# convert jsonl file structure
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        example = json.loads(line)
        chat_format = {
            "messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]}
            ]
        }
        outfile.write(json.dumps(chat_format) + "\n")

print(f"Converted data saved to {output_file}.")
