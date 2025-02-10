import requests
import json

api_key = os.getenv("OPEN_AI_API_KEY")

url = "https://api.openai.com/v1/fine_tuning/jobs"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "training_file": "file-VEaTeERKvnPsE3X7DHmzMY",  # uploaded file id
    "model": "gpt-4o-2024-08-06",  # model choice [needs to match jsonl format with role/user or input completion]
    "suffix": "pp_scouting_agent"  # suffix for naming
}


response = requests.post(url, headers=headers, data=json.dumps(payload))

# check response of request
if response.status_code == 200:
    print(f"Fine-tuning job created successfully. Job ID: {response.json()['id']}")
else:
    print(f"Error creating fine-tuning job: {response.status_code}")
    print(response.json())
