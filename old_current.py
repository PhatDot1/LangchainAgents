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
    print(f"[INFO]: {message}")

# === Helper to Ensure Field Values Are Strings ===
def ensure_string(val):
    if isinstance(val, list):
        return " ".join(map(str, val))
    return str(val)

# === Helper Functions for Programme Name Cleaning and Date Extraction ===
def clean_programme_name(name):
    """
    Removes codes in square brackets, quarter/year indicators (e.g. "24Q3"),
    time information, and trailing three-letter month abbreviations (e.g. "Sep")
    from a programme name. Returns the cleaned, human-friendly name.
    """
    name = ensure_string(name)
    name = re.sub(r'\[.*?\]', '', name)  # Remove codes in square brackets
    name = re.sub(r'\d{2}Q[1-4]', '', name)  # Remove quarter indicators
    name = re.sub(r'\b\d{1,2}(?::\d{2})?\s*(AM|PM|am|pm)\b', '', name)  # Remove time info
    name = re.sub(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', '', name, flags=re.IGNORECASE)  # Remove month abbreviations
    return name.strip()

def extract_quarter_info(text):
    """
    Extracts a quarter indicator (e.g., '24Q3') from the text and returns a tuple (year, quarter)
    where year is an integer (e.g. 2024) and quarter is an integer 1–4. Returns None if not found.
    """
    text = ensure_string(text)
    match = re.search(r'(\d{2})Q([1-4])', text)
    if match:
        year = int(match.group(1)) + 2000
        quarter = int(match.group(2))
        return (year, quarter)
    return None

# === Relevance Selection Using LLM ===
def select_relevant_events(events, user_prompt):
    """
    Uses the LLM to decide which events are most relevant given a list of candidate events and the user prompt.
    Each event is a tuple (field key, cleaned programme name, date_info).
    Returns a list of indices (0-indexed) of the selected events (maximum 2).
    """
    if not events:
        return []
    events_str = "\n".join([f"{i}: Field: {evt[0]}, Event: {evt[1]}, Date Info: {evt[2]}" for i, evt in enumerate(events)])
    prompt = f"""You are an expert at selecting relevant events from an Airtable record.
The record contains the following events:
{events_str}

The user prompt is: "{user_prompt}"

Please select the indices (0-indexed) of at most two events that are most relevant to the user's instructions.
Return the indices as a comma-separated list (for example: "0,2"). If no events are relevant, return an empty string.
"""
    log("Sending event selection prompt to LLM:")
    log(prompt)
    response = llm_chain([
        SystemMessage(content="You are an expert at selecting relevant events based on user instructions."),
        HumanMessage(content=prompt)
    ])
    answer = response.content.strip()
    log(f"LLM event selection response: {answer}")
    if not answer:
        return []
    try:
        indices = [int(x.strip()) for x in answer.split(",") if x.strip().isdigit()]
        return indices
    except Exception as e:
        log(f"Error parsing event selection: {str(e)}")
        return []

def extract_events(fields, keywords, exclusive=False):
    """
    Extracts events from the provided fields that match any of the given keywords.
    Each event is returned as a tuple: (field key, cleaned programme name, date_info).
    """
    events = []
    for key, value in fields.items():
        value_str = ensure_string(value)
        if any(kw in key.lower() for kw in keywords):
            cleaned = clean_programme_name(value_str)
            date_info = extract_quarter_info(value_str)
            events.append((key, cleaned, date_info))
    if len(events) > 2:
        selected = select_relevant_events(events, " ".join(keywords))
        if selected:
            events = [events[i] for i in selected if i < len(events)]
        else:
            events = events[:2]
    return events

# === Email Request Detection Helpers ===
def detect_email_components(prompt):
    """
    Returns a tuple (invitation_required, subject_required) based on whether the prompt
    mentions an email invitation and/or a subject line.
    """
    p = ensure_string(prompt).lower()
    invitation_required = "email invitation" in p or "invite" in p
    subject_required = "subject line" in p
    return invitation_required, subject_required

def is_email_request(prompt):
    """
    Returns True if the prompt appears to be related to email invitations.
    """
    invitation_required, subject_required = detect_email_components(prompt)
    return invitation_required or subject_required

def generate_subject_line(record):
    """
    Generates a concise subject line based on the record's context.
    The programme is taken from the Website URL for programme info.
    """
    control_fields = {"AI Prompt", "AI Description Destination Field", "AI Description Generator", "Full Record ID"}
    context_data = "\n".join(f"{key}: {ensure_string(value)}" 
                             for key, value in record.get("fields", {}).items() 
                             if key not in control_fields)
    programme_info = scrape_programme_url_field(record)
    if programme_info:
        programme_name = infer_programme_name(context_data, programme_info, "")
    else:
        programme_name = "the programme"
    prompt_text = f"""You are an expert at generating concise email subject lines.
Based on the following record context:
{context_data}

Generate a concise subject line for a targeted email regarding the programme "{programme_name}".
Return only the subject line.
"""
    log("Sending subject line generation prompt to LLM:")
    log(prompt_text)
    response = llm_chain([
        SystemMessage(content="You are an expert at generating concise email subject lines."),
        HumanMessage(content=prompt_text)
    ])
    subject_line = response.content.strip()
    return subject_line

def generic_response(record, user_prompt):
    """
    For non-email requests, generates a generic response using the record's context.
    Additionally, if the user's prompt contains any URLs, scrape them and append that information.
    """
    control_fields = {"AI Prompt", "AI Description Destination Field", "AI Description Generator", "Full Record ID"}
    context_data = "\n".join(f"{key}: {ensure_string(value)}" 
                             for key, value in record.get("fields", {}).items() 
                             if key not in control_fields)
    prompt_text = f"""Based on the following record context:
{context_data}

Answer the following question:
{user_prompt}
"""
    # Extract any URLs from the user's prompt
    urls_in_prompt = re.findall(r'(https?://\S+)', user_prompt)
    if urls_in_prompt:
        extra_scraped = ""
        for url in urls_in_prompt:
            if "github.com" in url:
                extra_scraped += f"\n[GitHub info for {url}]:\n{fetch_github_info(url)}\n"
            else:
                extra_scraped += f"\n[Website info for {url}]:\n{fetch_page_content(url)}\n"
        prompt_text += "\nAdditional Scraped Information from provided links:" + extra_scraped

    log("Sending generic response prompt to LLM:")
    log(prompt_text)
    response = llm_chain([
        SystemMessage(content="You are a helpful assistant that answers questions based on record context."),
        HumanMessage(content=prompt_text)
    ])
    return response.content.strip()

# === Airtable Functions ===
def fetch_records():
    """Fetch all records from Airtable where the 'AI Description Generator' is set to 'Generate'."""
    records = []
    offset = None
    log("Fetching records with 'AI Description Generator' set to 'Generate'...")
    while True:
        params = {"filterByFormula": "{AI Description Generator} = 'Generate'", "pageSize": 100}
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
    """Updates a record with the given fields."""
    allowed_values = {
        "Generated", "Error", "Please specify a destination table",
        "Cannot find destination field, plz check spelling", "Please specify a valid prompt"
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

# === External Tool Functions (Repeated for compatibility) ===
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
    """Scrapes a programme page for header and paragraph elements and returns their concatenated text (max 1000 characters)."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        elements = soup.find_all(["h1", "h2", "h3", "p"])
        texts = [el.get_text(strip=True) for el in elements if el.get_text(strip=True)]
        content = "\n".join(texts)
        if len(content) > 3000:
            content = content[:3000] + "..."
        return content
    except Exception as e:
        return f"Failed to fetch programme info from {url}: {str(e)}"

def fetch_github_info(url):
    """Fetches GitHub profile info, repositories and a README summary."""
    if not GITHUB_API_TOKEN:
        return "GitHub token not provided."
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_API_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    if not path_parts or len(path_parts) != 1:
        return "Invalid GitHub URL."
    username = path_parts[0]
    user_api_url = f"https://api.github.com/users/{username}"
    user_resp = requests.get(user_api_url, headers=headers)
    if user_resp.status_code != 200:
        return f"Failed to fetch GitHub user info: {user_resp.text}"
    user_info = user_resp.json()
    summary = (
        f"GitHub User Info:\nUsername: {user_info.get('login')}\nName: {user_info.get('name')}\n"
        f"Bio: {user_info.get('bio')}\nPublic repos: {user_info.get('public_repos')}\n"
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
    """Attempts a basic scrape of a LinkedIn page (may be limited due to authentication)."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text[:500]
    except Exception as e:
        return f"Failed to fetch LinkedIn info from {url}: {str(e)}"

def get_additional_tool_info_from_field(field_value):
    """Searches a given string for URLs and routes them to the appropriate external tool."""
    additional_info = ""
    field_value = ensure_string(field_value)
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
    """Extracts the signup link from the 'Website URL for programme info' field, if present."""
    fields = record.get("fields", {})
    url_field = fields.get("Website URL for programme info")
    if url_field:
        url_field = ensure_string(url_field)
        url = url_field.strip("[]'\"")
        if validators.url(url):
            return url
    return ""

def scrape_programme_url_field(record):
    """Checks for a programme info URL in the record and returns its scraped content."""
    fields = record.get("fields", {})
    url_field = fields.get("Website URL for programme info")
    if url_field:
        url_field = ensure_string(url_field)
        url = url_field.strip("[]'\"")
        if validators.url(url):
            return fetch_programme_info(url)
        else:
            return ""
    return ""

# === Supplemental Data Extraction & Constraints ===
def determine_exclusive_category(ai_prompt):
    """
    Returns "hackathon" if the prompt explicitly requests only hackathon participation (and not bootcamp),
    or "bootcamp" if only bootcamp participation is requested.
    """
    ai_lower = ensure_string(ai_prompt).lower()
    if "only" in ai_lower and "hackathon" in ai_lower and "bootcamp" not in ai_lower:
        return "hackathon"
    if "only" in ai_lower and "bootcamp" in ai_lower and "hackathon" not in ai_lower:
        return "bootcamp"
    return None

def extract_primary_hackathon(record):
    """
    Scans the record for hackathon fields and returns a cleaned hackathon name (e.g. "Particle Chain Abstraction Hackathon"),
    removing any codes, quarter indicators, trailing month abbreviations, and time info.
    """
    fields = record.get("fields", {})
    hackathon_keywords = ["hackathon", "hackathons"]
    for key, value in fields.items():
        value_str = ensure_string(value)
        if any(kw in key.lower() for kw in hackathon_keywords):
            return clean_programme_name(value_str)
    return None

def extract_events(fields, keywords, exclusive=False):
    """
    Extracts events from the provided fields that match any of the given keywords.
    Each event is returned as a tuple: (field key, cleaned programme name, date_info).
    """
    events = []
    for key, value in fields.items():
        value_str = ensure_string(value)
        if any(kw in key.lower() for kw in keywords):
            cleaned = clean_programme_name(value_str)
            date_info = extract_quarter_info(value_str)
            events.append((key, cleaned, date_info))
    return events

def select_relevant_events(events, user_prompt):
    """
    Uses the LLM to decide which events are most relevant given a list of candidate events and the user prompt.
    Each event is a tuple (field key, cleaned programme name, date_info).
    Returns a list of indices (0-indexed) of the selected events (maximum 2).
    """
    if not events:
        return []
    events_str = "\n".join([f"{i}: Field: {evt[0]}, Event: {evt[1]}, Date Info: {evt[2]}" for i, evt in enumerate(events)])
    prompt = f"""You are an expert at selecting relevant events from an Airtable record.
The record contains the following events:
{events_str}

The user prompt is: "{user_prompt}"

Please select the indices (0-indexed) of at most two events that are most relevant to the user's instructions.
Return the indices as a comma-separated list (for example: "0,2"). If no events are relevant, return an empty string.
"""
    log("Sending event selection prompt to LLM:")
    log(prompt)
    response = llm_chain([
        SystemMessage(content="You are an expert at selecting relevant events based on user instructions."),
        HumanMessage(content=prompt)
    ])
    answer = response.content.strip()
    log(f"LLM event selection response: {answer}")
    if not answer:
        return []
    try:
        indices = [int(x.strip()) for x in answer.split(",") if x.strip().isdigit()]
        return indices
    except Exception as e:
        log(f"Error parsing event selection: {str(e)}")
        return []

def extract_supplemental_data(record, ai_prompt):
    """
    Extracts supplemental data based on keywords in the prompt.
    If an exclusive category is requested (e.g. only hackathon), only that data is returned.
    If multiple events are found, the two most relevant (as determined by the LLM) are selected.
    """
    supplemental_info = ""
    fields = record.get("fields", {})
    bootcamp_keywords = ["bootcamp", "boot camp", "boot-camp"]
    hackathon_keywords = ["hackathon", "in-person event", "irl event", "in person", "event participation", "applied", "attended"]
    educate_keywords = ["educate", "education", "workshop", "seminar", "course"]
    ai_prompt_lower = ensure_string(ai_prompt).lower()
    exclusive = determine_exclusive_category(ai_prompt)
    events = []
    label = ""
    if exclusive == "hackathon":
        events = extract_events(fields, hackathon_keywords, exclusive=True)
        label = "Supplemental Hackathon Data"
    elif exclusive == "bootcamp":
        events = extract_events(fields, bootcamp_keywords, exclusive=True)
        label = "Supplemental Bootcamp Data"
    else:
        events = (extract_events(fields, bootcamp_keywords, exclusive=False)
                  + extract_events(fields, hackathon_keywords, exclusive=False)
                  + extract_events(fields, educate_keywords, exclusive=False))
        label = "Supplemental Events Data"
    if len(events) > 2:
        selected_indices = select_relevant_events(events, ai_prompt)
        if selected_indices:
            events = [events[i] for i in selected_indices if i < len(events)]
        else:
            events = events[:2]
    elif not events:
        events = []
    if events:
        event_strs = [f"{k}: {clean_programme_name(cleaned)}" for k, cleaned, date_info in events]
        supplemental_info += f"\n[{label}]:\n" + "\n".join(event_strs) + "\n"
    return supplemental_info

def extract_user_constraints(user_request):
    """
    Scans the user request for extra constraints (e.g. "five sentences max" or "include their GitHub").
    """
    constraints = []
    lower_req = ensure_string(user_request).lower()
    if "write an email" in lower_req:
        sentence_match = re.search(r'(\d+)\s*sentences', user_request, re.IGNORECASE)
        if sentence_match:
            num_sentences = sentence_match.group(1)
            constraints.append(f"Ensure the email is exactly {num_sentences} sentences long.")
        if re.search(r'include.*github', user_request, re.IGNORECASE):
            constraints.append("Include the user's GitHub information, if available.")
        if re.search(r'\bbrief\b', user_request, re.IGNORECASE):
            constraints.append("Keep the email brief.")
    return " ".join(constraints)

# === New: Generic and Subject Line Helpers ===
def fill_in_blank_template(record):
    """
    Processes a record in template mode using the record's own AI Prompt.
    """
    fields = record.get("fields", {})
    ai_prompt = ensure_string(fields.get("AI Prompt", ""))
    template = extract_email_template(ai_prompt)
    control_fields = {"AI Prompt", "AI Description Destination Field", "AI Description Generator", "Full Record ID"}
    context_data = "\n".join(f"{key}: {ensure_string(value)}" for key, value in fields.items() if key not in control_fields)
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
    if determine_exclusive_category(ai_prompt) == "hackathon":
        primary_hackathon = extract_primary_hackathon(record)
        if primary_hackathon:
            extra_info += f"\nWhen referencing past programme participation, mention only your participation in {primary_hackathon}."
    fill_prompt = f"""You are an expert email writer. The following is an email template with placeholders.
Fill in the blanks using the record context provided.
Revise any references to the original programme to reflect the new programme details.
Ensure the signup link appears immediately above the sign-off.
Do not include commentary; output only the completed email.

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
    log("Sending fill-in-the-blank prompt to LLM:")
    log(fill_prompt)
    response = llm_chain([
        SystemMessage(content="You are a helpful assistant that completes email templates and revises programme references."),
        HumanMessage(content=fill_prompt)
    ])
    completed_email = response.content.strip()
    log("LLM returned the completed email:")
    log(completed_email)
    return completed_email

def extract_email_template(ai_prompt):
    """
    If the AI prompt contains an email template in quotes, returns the text inside;
    otherwise, returns the full prompt.
    """
    match = re.search(r'"(.*?)"', ai_prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ai_prompt.strip()

def infer_programme_name(context_data, extra_info, ai_prompt):
    """
    Uses the provided context to deduce the new programme name.
    The programme names follow a format such as:
    "[E0602] 25Q1 Solana Rust Bootcamp 5pm Mar"
    Please ignore any codes in square brackets, quarter/year indicators, time information,
    and trailing month abbreviations. Optionally, if a year is relevant, include it naturally (e.g., "in 2024").
    IMPORTANT: The targeted programme must be determined from the Website URL for programme info.
    Return only the human-friendly programme name (e.g., "Solana Rust Bootcamp").
    """
    # For targeted programme, use the programme info scraped from the website URL.
    programme_info = scrape_programme_url_field({"fields": {"Website URL for programme info": extra_info}})
    inference_prompt = f"""You are an expert at deducing programme names from contextual data.
The programme names follow a format like: "[E0602] 25Q1 Solana Rust Bootcamp 5pm Mar". 
Please ignore any codes in square brackets, any quarter/year indicators (e.g. 25Q1, 24Q3), time information, and any trailing month abbreviations.
Optionally, if a year is relevant, include it naturally (e.g., "in 2024").
IMPORTANT: The targeted programme must be determined from the Website URL for programme info.
Return only the human-friendly programme name.

AI Prompt:
{ai_prompt}

Context:
{context_data}

Additional External Information:
{extra_info}
"""
    log("[INFO]: Inferring programme name from context...")
    log(inference_prompt)
    response = llm_chain([
        SystemMessage(content="You are a helpful assistant that extracts programme names."),
        HumanMessage(content=inference_prompt)
    ])
    inferred_programme_name = response.content.strip()
    if not inferred_programme_name:
        inferred_programme_name = "the event"
    log(f"[INFO]: Inferred Programme Name: {inferred_programme_name}")
    return inferred_programme_name

def extract_relevant_context(record):
    """
    Constructs the final prompt for full email generation using the record's AI Prompt.
    Adds extra user constraints if detected.
    If the prompt requests only hackathon participation, instructs the LLM to reference the specific hackathon.
    """
    control_fields = {"AI Prompt", "AI Description Destination Field", "AI Description Generator", "Full Record ID"}
    context_data = "\n".join(f"{key}: {ensure_string(value)}" 
                             for key, value in record.get("fields", {}).items() 
                             if key not in control_fields)
    extra_info = ""
    for key, value in record.get("fields", {}).items():
        if key not in control_fields and isinstance(value, str):
            extra_info += get_additional_tool_info_from_field(value)
    programme_scrape = scrape_programme_url_field(record)
    if programme_scrape:
        extra_info += "\n[Programme URL scrape]:\n" + programme_scrape
    ai_prompt = ensure_string(record.get("fields", {}).get("AI Prompt", ""))
    supplemental_info = extract_supplemental_data(record, ai_prompt)
    if supplemental_info:
        extra_info += supplemental_info
    signup_link = get_signup_link_from_record(record)
    inferred_programme = infer_programme_name(context_data, extra_info, ai_prompt)
    user_constraints = extract_user_constraints(ai_prompt)
    prompt_text = f"""You are an expert at crafting professional email invitations.
Using the following context, write a concise email invitation (no subject line) targeted to the new programme: {inferred_programme}.
Replace any references to the original programme with the new programme details.
Include the following signup link exactly as provided, and ensure it appears immediately above the sign-off:
{signup_link}
Do not output any placeholder tokens or templated text.
The email must conclude with the following sign-off exactly:

Best,
Encode

Context:
{context_data}

Additional External Information:
{extra_info}
"""
    if user_constraints:
        prompt_text += f"\nAdditional User Constraints: {user_constraints}\n"
    if determine_exclusive_category(ai_prompt) == "hackathon":
        primary_hackathon = extract_primary_hackathon(record)
        if primary_hackathon:
            prompt_text += f"\nWhen referencing your past programme participation, mention only your participation in {primary_hackathon}."
    prompt_text += "\nEmail Invitation:"
    log("Sending the following context extraction prompt to LLM:")
    log(prompt_text)
    response = llm_chain([
        SystemMessage(content="You are a helpful assistant that crafts professional email invitations and revises programme references."),
        HumanMessage(content=prompt_text)
    ])
    context_result = response.content.strip()
    log("LLM returned the following email invitation:")
    log(context_result)
    return context_result

# === Exclusive Gospel Tool ===
def email_gospel_tool(user_input: str) -> str:
    """
    Exclusive tool: If the record's AI Prompt starts with "GOD:", treat the remainder as gospel instructions.
    Generate an email invitation exactly as specified by the user's instructions while incorporating the record's context.
    Ensure that any programme names in the output are cleaned and that the targeted programme is determined
    solely from the Website URL for programme info field. Do not include a subject line.
    Additionally, if the user's instructions mention including GitHub info, incorporate that as well.
    """
    # Remove "GOD:" prefix from the user's input.
    gospel_instructions = user_input[4:].strip() if user_input.strip().upper().startswith("GOD:") else user_input

    # Fetch records and select the first record.
    records = fetch_records()
    if not records:
        return "No records found."
    record = records[0]
    fields = record.get("fields", {})

    # Build record context for personalization (excluding control fields).
    control_fields = {"AI Prompt", "AI Description Destination Field", "AI Description Generator", "Full Record ID"}
    context_data = "\n".join(
        f"{key}: {ensure_string(value)}"
        for key, value in fields.items()
        if key not in control_fields
    )

    # Scrape programme info from the Website URL for programme info field and clean it.
    raw_programme_info = scrape_programme_url_field(record)
    target_programme = clean_programme_name(raw_programme_info)
    if not target_programme:
        target_programme = "the programme"

    # Check if the user's instructions mention GitHub info.
    github_info = ""
    if "github" in gospel_instructions.lower():
        github_field = fields.get("Github")
        if github_field:
            github_info = fetch_github_info(ensure_string(github_field))

    # Build additional instructions.
    gospel_extra = (
        "IMPORTANT: When referencing programme names, remove any codes in square brackets, "
        "any quarter/year indicators (e.g., '24Q3', '25Q1'), any time information, and any trailing three-letter month abbreviations "
        "(e.g., 'Aug', 'Sep'). For example, '[E0602] 25Q1 Solana Rust Bootcamp 5pm Mar' should be rendered as 'Solana Rust Bootcamp', "
        "optionally including the year naturally if relevant."
    )
    gospel_subject = "Do not include any subject line in the output."

    # Construct the final prompt.
    prompt_text = f"""You are a strict email generator that follows user gospel instructions exactly.
User Gospel Instructions:
{gospel_instructions}

The targeted programme, determined from the Website URL for programme info, is: "{target_programme}".
Website URL: {get_signup_link_from_record(record)}
"""
    if github_info:
        prompt_text += f"\nInclude the following GitHub information for the user:\n{github_info}\n"
    prompt_text += f"""
Using the following record context for personalization:
{context_data}

{gospel_extra}
{gospel_subject}

Generate the email invitation exactly as specified, ensuring that the programme is referenced as "{target_programme}".
Return only the email invitation.
"""
    log("Sending gospel prompt to LLM:")
    log(prompt_text)
    response = llm_chain([
        SystemMessage(content="You are a strict email generator that follows user gospel instructions exactly."),
        HumanMessage(content=prompt_text)
    ])
    gospel_email = response.content.strip()
    log("LLM returned the gospel email invitation:")
    log(gospel_email)
    return gospel_email

# === Process Records Function ===
def process_records(records):
    """
    For each record, uses the record's own AI Prompt (from Airtable) to decide which email-generation tool to call,
    then updates the record accordingly. Processes each record independently.
    For prompts not clearly related to email invitations, a generic response is generated.
    """
    for record in records:
        fields = record.get("fields", {})
        record_id = record.get("id")
        ai_prompt = ensure_string(fields.get("AI Prompt", ""))
        destination_field_name = ensure_string(fields.get("AI Description Destination Field", ""))
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
            log(f"Processing record {record_id} using its stored AI Prompt.")
            # If the AI prompt starts with "GOD:", explicitly call the gospel tool.
            if ai_prompt.strip().upper().startswith("GOD:"):
                final_output = email_gospel_tool(ai_prompt)
            else:
                invitation_required, subject_required = detect_email_components(ai_prompt)
                if not invitation_required and not subject_required:
                    final_output = generic_response(record, ai_prompt)
                elif subject_required and not invitation_required:
                    final_output = generate_subject_line(record)
                elif invitation_required and subject_required:
                    subject = generate_subject_line(record)
                    mode = decide_email_mode(record, ai_prompt)
                    log(f"Email generation mode determined as: {mode}")
                    if mode == "TEMPLATE":
                        body = fill_in_blank_template(record)
                    else:
                        body = extract_relevant_context(record)
                    final_output = f"Subject: {subject}\n\n{body}"
                else:
                    mode = decide_email_mode(record, ai_prompt)
                    log(f"Email generation mode determined as: {mode}")
                    if mode == "TEMPLATE":
                        final_output = fill_in_blank_template(record)
                    else:
                        final_output = extract_relevant_context(record)
            signup_link = get_signup_link_from_record(record)
            if signup_link:
                idx = final_output.find("Best,")
                if idx != -1:
                    before = final_output[:idx].rstrip()
                    after = final_output[idx:]
                    if signup_link not in before:
                        final_output = before + f"\n\nSignup Link: {signup_link}\n\n" + after
                elif signup_link not in final_output:
                    final_output += f"\n\nSignup Link: {signup_link}"
            log(f"Final output for record {record_id}:\n{final_output}")
            update_result = update_record(record_id, {
                destination_field_name: final_output,
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

# --- Expose as a LangChain Tool ---
airtable_context_tool = Tool(
    name="AirtableRecordContextTool",
    func=airtable_record_context_tool,
    description="Extracts context from a single Airtable record (excluding control fields) and gathers external info from any URLs present."
)

# === Define Email Generation Tools for the Agent ===
def email_template_completion_tool(user_input: str) -> str:
    """
    Wraps the template completion branch.
    Uses the record's own AI Prompt from Airtable.
    """
    records = fetch_records()
    if not records:
        return "No records found."
    record = records[0]
    return fill_in_blank_template(record)

def email_full_generation_tool(user_input: str) -> str:
    """
    Wraps the full email generation branch.
    Uses the record's own AI Prompt from Airtable.
    """
    records = fetch_records()
    if not records:
        return "No records found."
    record = records[0]
    return extract_relevant_context(record)

# --- Tools List for the Agent ---
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

def is_email_request(prompt):
    """
    Returns True if the prompt appears to be related to email invitations.
    """
    p = ensure_string(prompt).lower()
    return any(keyword in p for keyword in ["email", "invitation", "subject line", "scouting"])

tools = [
    Tool(
        name="EmailTemplateCompletion",
        func=email_template_completion_tool,
        description="Completes an email template with placeholders. Use this tool if the Airtable record provides a template-style email prompt."
    ),
    Tool(
        name="EmailFullGeneration",
        func=email_full_generation_tool,
        description="Generates a full email invitation using record context and user constraints from the Airtable record. Use this tool if the record contains a complete email prompt."
    ),
    Tool(
        name="EmailGospel",
        func=email_gospel_tool,
        description="Exclusively generates an email invitation exactly as specified by the user when the AI Prompt starts with 'GOD:'. Follow the gospel instructions exactly."
    ),
    airtable_context_tool
]

# --- Set Up the Structured Chat Agent ---
prompt = hub.pull("hwchase17/structured-chat-agent")
llm = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, model="gpt-4")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# --- Initial System Message ---
initial_message = (
    "You are an AI assistant that generates tailored email invitations using available tools. "
    "For email invitations, use the prompt stored in the Airtable record's 'AI Prompt' field. "
    "If the record's prompt indicates a template, call EmailTemplateCompletion; if it is a complete prompt, call EmailFullGeneration; "
    "and if the prompt starts with 'GOD:', call EmailGospel (in which case, follow the instructions exactly). "
    "For queries unrelated to email invitations (for example, generating a subject line or answering simple questions), answer normally."
)
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# --- Process All Records Independently ---
if __name__ == "__main__":
    records_to_process = fetch_records()
    if not records_to_process:
        log("No records found.")
    else:
        for record in records_to_process:
            fields = record.get("fields", {})
            record_id = record.get("id")
            ai_prompt = ensure_string(fields.get("AI Prompt", ""))
            destination_field_name = ensure_string(fields.get("AI Description Destination Field", ""))
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
                log(f"Processing record {record_id} using its stored AI Prompt.")
                # If the AI prompt starts with "GOD:", explicitly call the gospel tool.
                if ai_prompt.strip().upper().startswith("GOD:"):
                    final_output = email_gospel_tool(ai_prompt)
                else:
                    invitation_required, subject_required = detect_email_components(ai_prompt)
                    if not invitation_required and not subject_required:
                        final_output = generic_response(record, ai_prompt)
                    elif subject_required and not invitation_required:
                        final_output = generate_subject_line(record)
                    elif invitation_required and subject_required:
                        subject = generate_subject_line(record)
                        mode = decide_email_mode(record, ai_prompt)
                        log(f"Email generation mode determined as: {mode}")
                        if mode == "TEMPLATE":
                            body = fill_in_blank_template(record)
                        else:
                            body = extract_relevant_context(record)
                        final_output = f"Subject: {subject}\n\n{body}"
                    else:
                        mode = decide_email_mode(record, ai_prompt)
                        log(f"Email generation mode determined as: {mode}")
                        if mode == "TEMPLATE":
                            final_output = fill_in_blank_template(record)
                        else:
                            final_output = extract_relevant_context(record)
                signup_link = get_signup_link_from_record(record)
                if signup_link:
                    idx = final_output.find("Best,")
                    if idx != -1:
                        before = final_output[:idx].rstrip()
                        after = final_output[idx:]
                        if signup_link not in before:
                            final_output = before + f"\n\nSignup Link: {signup_link}\n\n" + after
                    elif signup_link not in final_output:
                        final_output += f"\n\nSignup Link: {signup_link}"
                log(f"Final output for record {record_id}:\n{final_output}")
                update_result = update_record(record_id, {
                    destination_field_name: final_output,
                    "AI Description Generator": "Generated"
                })
                if update_result:
                    log(f"Output written to field '{destination_field_name}' for record {record_id}.")
                else:
                    log(f"Failed to write to destination field '{destination_field_name}' for record {record_id}.")
            except Exception as e:
                log(f"Error processing record {record_id}: {str(e)}")
                update_record(record_id, {"AI Description Generator": "Error"})
