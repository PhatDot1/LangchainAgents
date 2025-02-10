import os
import requests
from openai import OpenAI

# Airtable and OpenAI API setup
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
TABLE_NAME = os.environ.get("AIRTABLE_TABLE_NAME")
AIRTABLE_URL = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
}

client = OpenAI(api_key=OPEN_AI_API_KEY)

# logging function
def log(message):
    print(f"[INFO]: {message}")

# Fetch records meeting specific conditions
def fetch_records():
    records = []
    offset = None

    log("Fetching records with 'AI Description Generator' set to 'Generate'...")
    while True:
        params = {
            "filterByFormula": "{AI Description Generator} = 'Generate'",
            "fields[]": ["AI Prompt", "AI Description Destination Field"],
            "pageSize": 100
        }
        if offset:
            params["offset"] = offset

        response = requests.get(AIRTABLE_URL, headers=HEADERS, params=params)
        if response.status_code != 200:
            log(f"Error fetching records: {response.text}")
            break

        data = response.json()
        records.extend(data.get("records", []))
        offset = data.get("offset")

        if not offset:
            break

    log(f"Fetched {len(records)} records.")
    return records

def process_records(records):
    allowed_generator_values = {"Generated", "Error", "Please specify a destination table", "Cannot find destination field, plz check spelling"}

    for record in records:
        fields = record.get("fields", {})
        record_id = record.get("id")
        ai_prompt = fields.get("AI Prompt")
        destination_field_name = fields.get("AI Description Destination Field")  # Target field name

        log(f"Record ID: {record_id}")
        log(f"AI Prompt: {ai_prompt}")
        log(f"AI Description Destination Field: {destination_field_name}")

        if destination_field_name is None:
            log(f"Record {record_id} missing destination field value. Updating record...")
            update_record(record_id, {"AI Description Generator": "Please specify a destination table"})
            continue

        if not ai_prompt:
            log(f"Record {record_id} missing AI prompt. Updating record...")
            update_record(record_id, {"AI Description Generator": "Please specify a valid prompt"})
            continue

        #  generate and update the specified target field
        try:
            log(f"Processing record {record_id} with prompt: {ai_prompt}")

            response = client.chat.completions.create(
                model="ft:gpt-4o-2024-08-06:encode-club:pp-scouting-agent-t2:AsDekHHN",
                messages=[
                    {
                        "role": "user",
                        "content": ai_prompt
                    }
                ]
            )

            # Log entire API response
            log(f"Full API response: {response}")

            # Extract the generated content
            ai_output = response.choices[0].message.content.strip()
            log(f"Generated output for record {record_id}: {ai_output}")

            # Update target field with the response
            update_result = update_record(record_id, {
                destination_field_name: ai_output,  # Use value of "AI Description Destination Field" as target field
                "AI Description Generator": "Generated"
            })
            if update_result:
                log(f"Output written to field '{destination_field_name}' for record {record_id}.")
            else:
                log(f"Failed to write to destination field '{destination_field_name}' for record {record_id}.")
        except Exception as e:
            log(f"Error processing record {record_id}: {str(e)}")
            error_message = "Error" if "Error" in allowed_generator_values else "Error occurred"
            update_record(record_id, {"AI Description Generator": error_message})

def update_record(record_id, fields):
    for field, value in fields.items():
        if field == "AI Description Generator":
            allowed_values = {"Generated", "Error", "Please specify a destination table", "Cannot find destination field, plz check spelling"}
            if value not in allowed_values:
                log(f"Invalid value '{value}' for 'AI Description Generator'. Skipping update.")
                return False

    log(f"Attempting to update record {record_id} with fields: {fields}")
    url = f"{AIRTABLE_URL}/{record_id}"
    response = requests.patch(url, headers=HEADERS, json={"fields": fields})
    if response.status_code == 200:
        log(f"Record {record_id} updated successfully.")
        return True
    else:
        log(f"Failed to update record {record_id}: {response.text}")
        return False

# main
if __name__ == "__main__":
    records_to_process = fetch_records()
    process_records(records_to_process)
