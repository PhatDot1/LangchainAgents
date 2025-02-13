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
    encode.club or lu.ma). Scrapes the page for header and paragraph elements and returns
    their concatenated text (truncated to 1000 characters).
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
      2. Retrieves repositories and tallies language usage.
      3. Attempts to fetch a README from an appropriate repository.
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
    
    repos_api_url = f"https://api.github.com/users/{username}/repos"
    repos_resp = requests.get(repos_api_url, headers=headers)
    if repos_resp.status_code != 200:
        summary += f"\nFailed to fetch repositories: {repos_resp.text}"
        return summary
    repos = repos_resp.json()
    
    language_count = {}
    for repo in repos:
        lang = repo.get("language")
        if lang:
            language_count[lang] = language_count.get(lang, 0) + 1
    if language_count:
        total = sum(language_count.values())
        lang_summary = ", ".join(f"{lang}: {count/total*100:.1f}%" for lang, count in language_count.items())
        summary += f"\nLanguages breakdown: {lang_summary}\n"
    
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
    (LinkedIn pages are often behind authentication, so this may need further work.)
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
    Searches a given string for URLs and, if found, routes them to the appropriate external tool function.
    """
    additional_info = ""
    urls = re.findall(r'(https?://\S+)', field_value)
    for url in urls:
        if not validators.url(url):
            continue
        domain = urlparse(url).netloc.lower()
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

# === Supplemental Data Extraction Based on AI Prompt Keywords ===
def extract_supplemental_data(record, ai_prompt):
    """
    Scans record fields for supplemental data based on keywords found in the AI prompt.
    For example:
      - If the prompt mentions bootcamp participation, extract fields whose keys include "bootcamp".
      - If the prompt mentions hackathons or in-person events, extract those fields.
      - If the prompt mentions interests, skills, or location, extract those fields.
    Returns the supplemental info as a string to be appended to the final context.
    """
    supplemental_info = ""
    fields = record.get("fields", {})
    bootcamp_keywords = ["bootcamp", "boot camp", "boot-camp"]
    hackathon_keywords = ["hackathon", "in-person event", "irl event", "in person", "event participation", "applied", "attended"]
    interest_keywords = ["interest", "skill", "location"]

    ai_prompt_lower = ai_prompt.lower() if ai_prompt else ""

    # Bootcamp-related data
    if any(word in ai_prompt_lower for word in bootcamp_keywords):
        bootcamp_data = []
        for key, value in fields.items():
            if any(kw in key.lower() for kw in bootcamp_keywords):
                bootcamp_data.append(f"{key}: {value}")
        if bootcamp_data:
            supplemental_info += "\n[Supplemental Bootcamp Data]:\n" + "\n".join(bootcamp_data) + "\n"

    # Hackathon or in-person event data
    if any(word in ai_prompt_lower for word in hackathon_keywords):
        hackathon_data = []
        for key, value in fields.items():
            if any(kw in key.lower() for kw in hackathon_keywords):
                hackathon_data.append(f"{key}: {value}")
        if hackathon_data:
            supplemental_info += "\n[Supplemental Hackathon/IRL Event Data]:\n" + "\n".join(hackathon_data) + "\n"

    # Interests/skills/location data
    if any(word in ai_prompt_lower for word in interest_keywords):
        interest_data = []
        for key, value in fields.items():
            if any(kw in key.lower() for kw in interest_keywords):
                interest_data.append(f"{key}: {value}")
        if interest_data:
            supplemental_info += "\n[Supplemental Interests/Skills/Location Data]:\n" + "\n".join(interest_data) + "\n"

    return supplemental_info

# === New: Use the LLM to Interpret the User’s Intent ===
def decide_email_mode(record, user_request):
    """
    Uses the LLM to interpret the user’s request along with record context
    and decide whether the user is providing an email template (with blanks to fill in)
    or is asking for a completely generated email invitation.
    Expects the LLM to output one word: either 'TEMPLATE' or 'GENERATE'.
    """
    fields = record.get("fields", {})
    control_fields = {"AI Prompt", "AI Description Destination Field", "AI Description Generator"}
    # Build a simple context from non-control fields.
    context_data = "\n".join(f"{key}: {value}" for key, value in fields.items() if key not in control_fields)
    decision_prompt = f"""You are an expert at interpreting email invitation requests.
User Request:
{user_request}

Record Context:
{context_data}

Based on the above, decide whether the user is providing an email template (with blanks to fill in)
or is asking for a completely generated email invitation.
If the request contains an email template, output the single word: TEMPLATE.
Otherwise, output the single word: GENERATE.
Output only one word.
"""
    log("Sending decision prompt to LLM:")
    log(decision_prompt)
    response = llm_chain([
         SystemMessage(content="You are an assistant that determines email invitation modes."),
         HumanMessage(content=decision_prompt)
    ])
    decision = response.content.strip().upper()
    log(f"Decision from LLM: {decision}")
    if "TEMPLATE" in decision:
         return "TEMPLATE"
    else:
         return "GENERATE"

# === Fill-in-the-Blank Email Template Completion ===
def extract_email_template(ai_prompt):
    """
    If the AI prompt contains an email template in quotes, extract and return the text inside the quotes.
    Otherwise, return the full prompt.
    """
    match = re.search(r'"(.*?)"', ai_prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ai_prompt.strip()

def fill_in_blank_template(record):
    """
    Processes a record whose AI prompt is a fill-in-the-blank email template.
    It extracts the template from the prompt, gathers all context (including external info,
    supplemental data, programme info, signup link, and inferred programme name),
    and instructs the LLM to fill in the blanks.
    Additionally, the prompt instructs the LLM to revise any references to the original programme
    so that they reflect the new programme details.
    Importantly, ensure that the signup link appears immediately above the sign-off.
    """
    fields = record.get("fields", {})
    ai_prompt = fields.get("AI Prompt", "")
    template = extract_email_template(ai_prompt)
    control_fields = {"AI Prompt", "AI Description Destination Field", "AI Description Generator"}
    context_data = "\n".join(f"{key}: {value}" for key, value in fields.items() if key not in control_fields)
    extra_info = ""
    for key, value in fields.items():
        if key not in control_fields and isinstance(value, str):
            extra_info += get_additional_tool_info_from_field(value)
    programme_scrape = scrape_programme_url_field(record)
    if programme_scrape:
        extra_info += "\n[Programme URL scrape]:\n" + programme_scrape
    supplemental_info = extract_supplemental_data(record, ai_prompt)
    if supplemental_info:
        extra_info += supplemental_info
    signup_link = get_signup_link_from_record(record)
    inferred_programme = infer_programme_name(context_data, extra_info, ai_prompt)
    
    fill_prompt = f"""You are an expert email writer. The following is an email template with placeholders (indicated by curly braces).
Fill in the blanks and adjust the email based on the record context provided.
Important: Revise any references to the original programme so that they reflect the new programme details as indicated below.
Also, ensure that the signup link is included and appears immediately above the sign-off.
Do not include any commentary or explanations; output only the completed email.

Email Template:
{template}

Record Context:
{context_data}

Additional External Information:
{extra_info}

Signup Link: {signup_link}

New Programme Details: {inferred_programme}

Completed Email:
"""
    log("Sending fill-in-the-blank prompt to the LLM:")
    log(fill_prompt)
    
    response = llm_chain([
         SystemMessage(content="You are a helpful assistant that completes email templates and revises programme references."),
         HumanMessage(content=fill_prompt)
    ])
    completed_email = response.content.strip()
    log("LLM returned the completed email:")
    log(completed_email)
    return completed_email

# === New: Infer Programme Name Dynamically ===
def infer_programme_name(context_data, extra_info, ai_prompt):
    """
    Uses the provided context to deduce the programme name.
    Returns the inferred programme name in a single line.
    """
    inference_prompt = f"""You are an expert at deducing programme names from contextual data.
Based on the following information, extract the correct programme name in a single line.

AI Prompt:
{ai_prompt}

Context:
{context_data}

Additional External Information:
{extra_info}

Return only the programme name.
"""
    log("[INFO]: Inferring programme name from context...")
    response = llm_chain([
        SystemMessage(content="You are a helpful assistant that extracts programme names."),
        HumanMessage(content=inference_prompt)
    ])
    inferred_programme_name = response.content.strip()
    if not inferred_programme_name:
        inferred_programme_name = "the event"
    log(f"[INFO]: Inferred Programme Name: {inferred_programme_name}")
    return inferred_programme_name

# === Final Context Extraction and Prompt Building for New Email Generation ===
def extract_relevant_context(record):
    """
    Builds the context for the record by combining its non-control fields and any additional
    external info fetched via URLs. Also extracts the signup link, supplements the context based
    on keywords in the AI prompt, and dynamically infers the programme name.
    Then it instructs the LLM to craft a professional email invitation.
    IMPORTANT: The prompt instructs the LLM to revise any references to the original programme
    so that they reflect the new programme (as determined by the signup link and scraped info),
    and to ensure that the signup link appears immediately above the sign-off.
    """
    fields = record.get("fields", {})
    control_fields = {"AI Prompt", "AI Description Destination Field", "AI Description Generator"}
    context_data = "\n".join(f"{key}: {value}" for key, value in fields.items() if key not in control_fields)
    
    extra_info = ""
    for key, value in fields.items():
        if key not in control_fields and isinstance(value, str):
            extra_info += get_additional_tool_info_from_field(value)
    
    programme_scrape = scrape_programme_url_field(record)
    if programme_scrape:
        extra_info += "\n[Programme URL scrape]:\n" + programme_scrape

    ai_prompt = fields.get("AI Prompt", "")
    supplemental_info = extract_supplemental_data(record, ai_prompt)
    if supplemental_info:
        extra_info += supplemental_info
    
    signup_link = get_signup_link_from_record(record)
    inferred_programme = infer_programme_name(context_data, extra_info, ai_prompt)
    
    prompt_text = f"""You are an expert at crafting professional email invitations.
Using the following context, write a concise email invitation body (no subject line) targeted to the new programme: {inferred_programme}.
Ensure that any references to the original programme are replaced with the new programme details.
Include the following signup link exactly as provided, and ensure that it appears immediately above the sign-off:
{signup_link}
Do not output any placeholder tokens or templated text.
Output only the email body text with no extra labels or commentary.
The email must conclude with the following sign-off exactly:

Best,
Encode

Context:
{context_data}

Additional External Information:
{extra_info}

Email Invitation:
"""
    log("Sending the following context extraction prompt to the LLM:")
    log(prompt_text)
    
    response = llm_chain([
        SystemMessage(content="You are a helpful assistant that crafts professional email invitations and revises programme references."),
        HumanMessage(content=prompt_text)
    ])
    
    context_result = response.content.strip()
    log("LLM returned the following email invitation:")
    log(context_result)
    return context_result

# === Record Processing ===
def process_records(records):
    """
    For each record, verifies required fields, extracts context (including external info),
    uses the LLM to decide whether to complete a provided email template or to generate a new email,
    obtains the generated output from the appropriate API call, and updates the record.
    Regardless of the mode, ensures that the relevant Signup Link is integrated above the sign-off.
    """
    for record in records:
        fields = record.get("fields", {})
        record_id = record.get("id")
        ai_prompt = fields.get("AI Prompt")
        destination_field_name = fields.get("AI Description Destination Field")
        
        log(f"Record ID: {record_id}")
        log(f"AI Prompt: {ai_prompt}")
        log(f"AI Description Destination Field: {destination_field_name}")
        
        if not destination_field_name:
            log(f"Record {record_id} missing destination field value. Updating record...")
            update_record(record_id, {"AI Description Generator": "Please specify a destination table"})
            continue
        
        if not ai_prompt:
            log(f"Record {record_id} missing AI prompt. Updating record...")
            update_record(record_id, {"AI Description Generator": "Please specify a valid prompt"})
            continue
        
        try:
            log(f"Processing record {record_id} with prompt: {ai_prompt}")
            # Use the LLM to decide which approach to use
            mode = decide_email_mode(record, ai_prompt)
            log(f"Email generation mode determined as: {mode}")
            if mode == "TEMPLATE":
                final_email = fill_in_blank_template(record)
            else:
                final_email = extract_relevant_context(record)
            
            # Always ensure the Signup Link is included above the sign-off.
            signup_link = get_signup_link_from_record(record)
            if signup_link:
                # Check for a sign-off marker (e.g., "Best,")
                idx = final_email.find("Best,")
                if idx != -1:
                    # Insert the signup link above the sign-off
                    before = final_email[:idx].rstrip()
                    after = final_email[idx:]
                    if signup_link not in before:
                        final_email = before + f"\n\nSignup Link: {signup_link}\n\n" + after
                elif signup_link not in final_email:
                    final_email += f"\n\nSignup Link: {signup_link}"
            
            log(f"Final email invitation for record {record_id}:\n{final_email}")
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

# Expose as a LangChain Tool if desired.
airtable_context_tool = Tool(
    name="AirtableRecordContextTool",
    func=airtable_record_context_tool,
    description="Extracts context from a single Airtable record (excluding control fields) and gathers external info from any URLs present."
)

if __name__ == "__main__":
    records_to_process = fetch_records()
    process_records(records_to_process)
