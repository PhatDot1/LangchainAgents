import requests

api_key = os.getenv("OPEN_AI_API_KEY")

job_id = "ftjob-sb2gX83Jy3uLgzg7TOvHD3WO"  # job id
url = f"https://api.openai.com/v1/fine_tuning/jobs/{job_id}"

headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(url, headers=headers)

# check request response
if response.status_code == 200:
    job_status = response.json()
    print("Fine-Tuning Job Status:")
    print(f"ID: {job_status['id']}")
    print(f"Model: {job_status['model']}")
    print(f"Status: {job_status['status']}")
    print(f"Created At: {job_status['created_at']}")
    print(f"Fine-Tuned Model: {job_status.get('fine_tuned_model', 'Not available yet')}")
else:
    print(f"Error checking job status: {response.status_code}")
    print(response.json())
