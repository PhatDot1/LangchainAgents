import requests


api_key = os.getenv("OPEN_AI_API_KEY")
file_path = "1_chat.jsonl"
file_name = "1_chat.jsonl"
purpose = "fine-tune"


url = "https://api.openai.com/v1/files"
headers = {
    "Authorization": f"Bearer {api_key}"
}
files = {
    "file": (file_name, open(file_path, "rb"), "application/jsonl")
}
data = {
    "purpose": purpose
}

response = requests.post(url, headers=headers, files=files, data=data)

if response.status_code == 200:
    upload_id = response.json()["id"]
    print(f"File uploaded successfully: {upload_id}")
else:
    print(f"Error uploading file: {response.text}")
