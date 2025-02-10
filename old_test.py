import os
import re
import requests
import validators
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import base64
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.tools import Tool
from dotenv import load_dotenv

load_dotenv()

# === Airtable & API Credentials ===
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
TABLE_NAME = os.environ.get("AIRTABLE_TABLE_NAME")
AIRTABLE_URL = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
}

OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
GITHUB_API_TOKEN = os.environ.get("GITHUB_API_KEY")  # Your GitHub token

client = OpenAI(api_key=OPEN_AI_API_KEY)
llm_chain = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, model="gpt-4")

# === Utility Logging ===
def log(message):
    """Verbose logging to the terminal."""
    print(f"[INFO]: {message}")

# === Airtable Functions ===
def fetch_records():
    """
    Fetch all records from Airtable where the 'AI Description Generator'
    is set to 'Generate'.
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
    Updates a record with the given fields.
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

# === External Tool Functions ===
def fetch_page_content(url):
    """Fetch and return the text content of a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except requests.RequestException as e:
        return f"Failed to fetch {url}: {str(e)}"

def fetch_programme_info(url):
    """
    Fetches details from an intended outreach programme page (e.g. from domains like
    encode.club or lu.ma). This version scrapes the page for all header (h1, h2, h3)
    and paragraph (p) elements and returns their concatenated text, truncated to 1000 characters.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        elements = soup.find_all(["h1", "h2", "h3", "p"])
        texts = [el.get_text(strip=True) for el in elements if el.get_text(strip=True)]
        content = "\n".join(texts)
        if len(content) > 1000:
            content = content[:1000] + "..."
        return content
    except Exception as e:
        return f"Failed to fetch programme info from {url}: {str(e)}"

def fetch_github_info(url):
    """
    Given a GitHub user URL (e.g. https://github.com/JohnDoe), this function:
      1. Fetches the user’s profile info (including bio).
      2. Retrieves all repositories from https://api.github.com/users/JohnDoe/repos.
      3. Tallies the languages used (calculating a percentage breakdown).
      4. Checks for a repository with a name that exactly matches the username.
         If found, it fetches its README; otherwise, it falls back to selecting the
         repository with the highest star count.
    Returns a formatted summary.
    """
    if not GITHUB_API_TOKEN:
        return "GitHub token not provided."
    
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_API_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    if not path_parts:
        return "Invalid GitHub URL."
    
    if len(path_parts) != 1:
        return "Please provide a GitHub user URL (e.g. https://github.com/JohnDoe)."
    
    username = path_parts[0]
    
    # 1. Fetch user profile info.
    user_api_url = f"https://api.github.com/users/{username}"
    user_resp = requests.get(user_api_url, headers=headers)
    if user_resp.status_code != 200:
        return f"Failed to fetch GitHub user info: {user_resp.text}"
    user_info = user_resp.json()
    summary = (
        f"GitHub User Info:\n"
        f"Username: {user_info.get('login')}\n"
        f"Name: {user_info.get('name')}\n"
        f"Bio: {user_info.get('bio')}\n"
        f"Public repos: {user_info.get('public_repos')}\n"
    )
    
    # 2. Fetch all repositories.
    repos_api_url = f"https://api.github.com/users/{username}/repos"
    repos_resp = requests.get(repos_api_url, headers=headers)
    if repos_resp.status_code != 200:
        summary += f"\nFailed to fetch repositories: {repos_resp.text}"
        return summary
    repos = repos_resp.json()
    
    # 3. Tally languages.
    language_count = {}
    for repo in repos:
        lang = repo.get("language")
        if lang:
            language_count[lang] = language_count.get(lang, 0) + 1
    if language_count:
        total = sum(language_count.values())
        lang_summary = ", ".join(f"{lang}: {count/total*100:.1f}%" for lang, count in language_count.items())
        summary += f"\nLanguages breakdown: {lang_summary}\n"
    
    # 4. Determine which repository to use for README.
    target_repo = None
    for repo in repos:
        if repo.get("name", "").lower() == username.lower():
            target_repo = repo
            break
    if not target_repo:
        top_repo = None
        top_stars = -1
        for repo in repos:
            stars = repo.get("stargazers_count", 0)
            if stars > top_stars:
                top_stars = stars
                top_repo = repo
        target_repo = top_repo
    
    # 5. Attempt to fetch the README.
    if target_repo:
        owner = target_repo.get("owner", {}).get("login")
        repo_name = target_repo.get("name")
        readme_api_url = f"https://api.github.com/repos/{owner}/{repo_name}/readme"
        readme_resp = requests.get(readme_api_url, headers=headers)
        if readme_resp.status_code == 200:
            readme_json = readme_resp.json()
            try:
                readme_content = base64.b64decode(readme_json.get("content", "")).decode("utf-8", errors="ignore")
                summary += f"\nREADME from repo ({repo_name}):\n{readme_content[:500]}...\n"
            except Exception as e:
                summary += f"\nError decoding README: {str(e)}\n"
        else:
            summary += f"\nFailed to fetch README for repo '{repo_name}': {readme_resp.text}\n"
    else:
        summary += "\nNo repositories found for this user.\n"
    
    return summary

def fetch_linkedin_info(url):
    """
    For LinkedIn URLs, attempt a basic scrape.
    (Note: LinkedIn pages are often behind authentication, so this may need further work.)
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text[:500]
    except Exception as e:
        return f"Failed to fetch LinkedIn info from {url}: {str(e)}"

def get_additional_tool_info_from_field(field_value):
    """
    Searches a given string (or list) for URLs and, if found, routes them to the appropriate external tool function.
    Recognizes GitHub and LinkedIn URLs as before. For URLs from intended outreach domains (e.g.
    encode.club or lu.ma), it calls fetch_programme_info to extract detailed page content.
    Additionally, URLs from unwanted domains (e.g. amazonaws.com or s3) are ignored.
    """
    additional_info = ""
    if isinstance(field_value, list):
        if field_value:
            field_value = field_value[0]
        else:
            return additional_info
    field_value = field_value.strip("[]'\"")
    
    urls = re.findall(r'(https?://\S+)', field_value)
    for url in urls:
        if not validators.url(url):
            continue
        domain = urlparse(url).netloc.lower()
        if "amazonaws.com" in domain or "s3" in domain:
            continue
        if "github.com" in domain:
            info = fetch_github_info(url)
            additional_info += f"\n[GitHub info for {url}]:\n{info}\n"
        elif "linkedin.com" in domain:
            info = fetch_linkedin_info(url)
            additional_info += f"\n[LinkedIn info for {url}]:\n{info}\n"
        elif "encode.club" in domain or "lu.ma" in domain:
            info = fetch_programme_info(url)
            additional_info += f"\n[Programme info for {url}]:\n{info}\n"
        else:
            info = fetch_page_content(url)
            additional_info += f"\n[Website info for {url}]:\n{info[:500]}...\n"
    return additional_info

def get_signup_link_from_record(record):
    """
    Extracts and returns the clean signup link from the "Website URL for programme info" field,
    if present.
    """
    fields = record.get("fields", {})
    url_field = fields.get("Website URL for programme info")
    if url_field:
        if isinstance(url_field, list) and url_field:
            url = url_field[0]
        else:
            url = str(url_field)
        url = url.strip("[]'\"")
        if validators.url(url):
            return url
    return ""

def scrape_programme_url_field(record):
    """
    Checks if the record has a field called "Website URL for programme info".
    If found, cleans the URL and returns the scraped programme info using fetch_programme_info.
    """
    fields = record.get("fields", {})
    url_field = fields.get("Website URL for programme info")
    if url_field:
        if isinstance(url_field, list) and url_field:
            url = url_field[0]
        else:
            url = str(url_field)
        url = url.strip("[]'\"")
        if validators.url(url):
            return fetch_programme_info(url)
        else:
            return ""
    return ""

def extract_relevant_context(record):
    """
    Builds the context for the record by combining its non-control fields and any additional
    external info fetched via URLs. Also extracts the signup link from the programme URL field.
    """
    fields = record.get("fields", {})
    control_fields = {"AI Prompt", "AI Description Destination Field", "AI Description Generator"}
    
    context_data = "\n".join(
        f"{key}: {value}" for key, value in fields.items() if key not in control_fields
    )
    
    extra_info = ""
    for key, value in fields.items():
        if key not in control_fields and isinstance(value, str):
            extra_info += get_additional_tool_info_from_field(value)
    
    # Scrape the programme URL field and add its content.
    programme_scrape = scrape_programme_url_field(record)
    if programme_scrape:
        extra_info += "\n[Programme URL scrape]:\n" + programme_scrape
    
    # Extract the signup link.
    signup_link = get_signup_link_from_record(record)
    
    # Build the final prompt with strict instructions.
    prompt_text = f"""You are an expert at crafting professional email invitations.
Using the following context, write a concise email invitation body (no subject line) to the Encode AI London Hackathon.
Include the signup link exactly as provided below.
Do not output any placeholder tokens or templated text (e.g. no "{{...}}" or "[Insert Signup link here]").
Output only the email body text with no extra labels or commentary.
The email must conclude exactly with:

Best,
Encode

Signup link: {signup_link}

Context:
{context_data}

Additional external information gathered:
{extra_info}

Email Invitation:
"""
    log("Sending the following context extraction prompt to the LLM:")
    log(prompt_text)
    
    response = llm_chain([
        SystemMessage(content="You are a helpful assistant that crafts professional email invitations."),
        HumanMessage(content=prompt_text)
    ])
    
    context_result = response.content.strip()
    log("LLM returned the following email invitation:")
    log(context_result)
    return context_result

def process_records(records):
    """
    For each record, verifies required fields, extracts context (including external info),
    builds the final prompt, gets the generated output from the first API call,
    and updates the record.
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
            
            # Extract the context and the final email invitation.
            final_email = extract_relevant_context(record)
            log(f"Final email invitation for record {record_id}:\n{final_email}")
            
            # Use the generated email invitation directly.
            ai_output = final_email
            
            log(f"Updating record {record_id} with the following email invitation:\n{ai_output}")
            
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

def airtable_record_context_tool():
    records = fetch_records()
    if not records:
        return "No records found."
    return extract_relevant_context(records[0])

airtable_context_tool = Tool(
    name="AirtableRecordContextTool",
    func=airtable_record_context_tool,
    description="Extracts relevant context from a single Airtable record (excluding control fields) and gathers external info from any URLs present."
)

if __name__ == "__main__":
    records_to_process = fetch_records()
    process_records(records_to_process)
