import os
import requests
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.tools import Tool


# credentials and base/table info
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
TABLE_NAME = os.environ.get("AIRTABLE_TABLE_NAME")
AIRTABLE_URL = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
}

client = OpenAI(api_key=OPEN_AI_API_KEY)

OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
llm_chain = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, model="gpt-4")



def log(message):
    """Verbose logging to the terminal."""
    print(f"[INFO]: {message}")



def fetch_records():
    """
    Fetches all records from Airtable where the 'AI Description Generator'
    field is set to 'Generate'. All fields for each record are returned.
    """
    records = []
    offset = None

    log("Fetching records with 'AI Description Generator' set to 'Generate'...")
    while True:
        params = {
            "filterByFormula": "{AI Description Generator} = 'Generate'",
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

def update_record(record_id, fields):
    """
    Updates a record with the given fields. For the control field 'AI Description Generator',
    only allowed values are accepted.
    """
    allowed_values = {
        "Generated",
        "Error",
        "Please specify a destination table",
        "Cannot find destination field, plz check spelling",
        "Please specify a valid prompt"
    }
    for field, value in fields.items():
        if field == "AI Description Generator" and value not in allowed_values:
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

def extract_relevant_context(record):
    """
    Given a record, uses its fields (excluding control fields) and its AI prompt to
    ask an LLM which pieces of data are most relevant for generating a detailed description.
    """
    fields = record.get("fields", {})
    # control fields that should be excluded
    control_fields = {"AI Prompt", "AI Description Destination Field", "AI Description Generator"}
    
    # build string of field names and their values for the non-control fields
    context_data = "\n".join(f"{key}: {value}" for key, value in fields.items() if key not in control_fields)
    
    # get record's prompt
    ai_prompt = fields.get("AI Prompt", "")
    
    # prep prompt for the LLM to extract the relevant context from the record's data
    prompt_text = f"""You are an expert at extracting relevant context from structured data.
The record's additional fields (key-value pairs) are:
{context_data}

The record's AI prompt is:
{ai_prompt}

Please extract and return only the key pieces of context (include the field names and their values) 
that are most relevant for generating a detailed description in response to the prompt.
Return your answer as a concise text summary.
"""
    log("Sending the following context extraction prompt to the LLM:")
    log(prompt_text)
    
    response = llm_chain([
        SystemMessage(content="You are a helpful assistant that extracts relevant context."),
        HumanMessage(content=prompt_text)
    ])
    
    context_result = response.content.strip()
    log("LLM returned the following relevant context:")
    log(context_result)
    return context_result

def process_records(records):
    """
    For each fetched record, this function:
      - Verifies that required fields exist.
      - Extracts context from the record's own fields based on its AI prompt.
      - Combines the extracted context with the original prompt.
      - Sends the final prompt to the OpenAI API.
      - Updates the record with the generated output.
    """
    for record in records:
        fields = record.get("fields", {})
        record_id = record.get("id")
        ai_prompt = fields.get("AI Prompt")
        destination_field_name = fields.get("AI Description Destination Field")

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

        try:
            log(f"Processing record {record_id} with prompt: {ai_prompt}")
            
            # extract context from the records non-control fields based on its prompt.
            extracted_context = extract_relevant_context(record)
            
            # build final prompt by combining the extracted context with the original prompt.
            final_prompt = f"Context:\n{extracted_context}\n\nPrompt:\n{ai_prompt}"
            log(f"Final prompt for record {record_id}:\n{final_prompt}")
            
            # call OpenAI chat completions endpoint with the final prompt.
            response = client.chat.completions.create(
                model="ft:gpt-4o-2024-08-06:encode-club:pp-scouting-agent-t2:AsDekHHN",
                messages=[
                    {"role": "user", "content": final_prompt}
                ]
            )
            log(f"Full API response: {response}")
            
            ai_output = response.choices[0].message.content.strip()
            log(f"Generated output for record {record_id}: {ai_output}")
            
            update_result = update_record(record_id, {
                destination_field_name: ai_output,
                "AI Description Generator": "Generated"
            })
            if update_result:
                log(f"Output written to field '{destination_field_name}' for record {record_id}.")
            else:
                log(f"Failed to write to destination field '{destination_field_name}' for record {record_id}.")
        except Exception as e:
            log(f"Error processing record {record_id}: {str(e)}")
            update_record(record_id, {"AI Description Generator": "Error"})

# optionally to wrap the context extraction as a LangChain Tool:
def airtable_record_context_tool():
    records = fetch_records()
    if not records:
        return "No records found."
    # extract context from the first record.
    return extract_relevant_context(records[0])

airtable_context_tool = Tool(
    name="AirtableRecordContextTool",
    func=airtable_record_context_tool,
    description="Extracts relevant context from a single Airtable record (excluding control fields) based on its prompt."
)


if __name__ == "__main__":
    # 1. fetch records (with all fields)
    records_to_process = fetch_records()
    
    # 2. process each record by extracting record-specific context and generating a description.
    process_records(records_to_process)
