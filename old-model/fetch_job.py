import requests

api_key = os.getenv("OPEN_AI_API_KEY")

job_id = "ftjob-e0GUftOKJGZBny9EWlI6ry2F"  # job id
events_url = f"https://api.openai.com/v1/fine_tuning/jobs/{job_id}/events"

headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(events_url, headers=headers)

# check response
if response.status_code == 200:
    events = response.json()["data"]
    print("Fine-Tuning Job Events:")
    for event in events:
        print(f"- {event['created_at']}: {event['message']}")
else:
    print(f"Error fetching job events: {response.status_code}")
    print(response.json())
