from dotenv import load_dotenv
import os
import requests
import json
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Get API keys
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
GITHUB_API_KEY = os.environ.get("GITHUB_API_KEY")

# Initialize LLM
llm = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, model="gpt-4")

# Airtable headers
AIRTABLE_HEADERS = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

### **Extract Record ID Dynamically from Input** ###
def extract_record_id(user_query):
    """Uses LLM to extract the Airtable record ID from a natural query."""
    extraction_prompt = f"""
    User provided this query: "{user_query}"

    Your job is to extract the Airtable record ID from the query.

    - An Airtable record ID always starts with 'rec' and consists of letters/numbers.
    - If no record ID is found, return "None".

    Example Outputs:
    - Input: "get details for rec1234" → Output: "rec1234"
    - Input: "find info on recXYZ567" → Output: "recXYZ567"
    - Input: "who is John Doe?" → Output: "None"

    Extract only the record ID as a plain string.
    """

    record_id = llm.invoke(extraction_prompt).strip()
    return record_id if record_id.startswith("rec") else None

### **Fetch All Airtable Bases** ###
def get_all_bases():
    """Fetches all bases available via the Airtable API."""
    url = "https://api.airtable.com/v0/meta/bases"
    response = requests.get(url, headers=AIRTABLE_HEADERS)
    return response.json().get("bases", []) if response.status_code == 200 else []

### **Fetch All Tables in a Base** ###
def get_all_tables(base_id):
    """Fetches all tables in a given base."""
    url = f"https://api.airtable.com/v0/meta/bases/{base_id}/tables"
    response = requests.get(url, headers=AIRTABLE_HEADERS)
    return response.json().get("tables", []) if response.status_code == 200 else []

### **Find the Most Relevant Table Using LLM** ###
def find_relevant_table(query):
    """Uses LLM to determine the best table for a given query."""
    bases = get_all_bases()
    tables_info = {}

    for base in bases:
        tables = get_all_tables(base["id"])
        for table in tables:
            tables_info[table["id"]] = {
                "base_id": base["id"],
                "table_name": table["name"],
                "fields": [f["name"] for f in table["fields"]],
            }

    decision_prompt = f"""
    Available Airtable tables:

    {json.dumps(tables_info, indent=2)}

    Based on the user query: "{query}", which table is most relevant?
    Return ONLY the table_id.
    """
    selected_table_id = llm.invoke(decision_prompt).strip()

    return tables_info.get(selected_table_id, {})

### **Search for Record in Airtable** ###
def search_airtable_record(record_id, query):
    """Finds a record using LLM to determine the best table."""
    table_info = find_relevant_table(query)
    if not table_info:
        return None, "No relevant table found."

    base_id, table_id = table_info["base_id"], table_info["table_name"]
    url = f"https://api.airtable.com/v0/{base_id}/{table_id}/{record_id}"
    response = requests.get(url, headers=AIRTABLE_HEADERS)

    return response.json().get("fields", {}), table_info["table_name"] if response.status_code == 200 else (None, "Record not found.")

### **Extract Useful Fields with LLM** ###
def extract_useful_fields(fields, query):
    """Uses LLM to extract the most relevant fields from a record."""
    extraction_prompt = f"""
    Here is an Airtable record:

    {json.dumps(fields, indent=2)}

    The user is looking for: "{query}". 
    Which fields provide the most useful information?
    
    Return a JSON object with only the most relevant fields.
    """
    return json.loads(llm.invoke(extraction_prompt))

### **Fetch GitHub Data (if applicable)** ###
def fetch_github_data(github_url):
    """Uses GitHub API to fetch user details and repos."""
    username = github_url.rstrip("/").split("/")[-1]
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_API_KEY}",
    }

    profile_url = f"https://api.github.com/users/{username}"
    profile_response = requests.get(profile_url, headers=headers).json()

    repos_url = f"https://api.github.com/users/{username}/repos"
    repos_response = requests.get(repos_url, headers=headers)

    repo_languages = {}
    total_commits = 0

    if repos_response.status_code == 200:
        repos = repos_response.json()
        for repo in repos:
            lang_response = requests.get(repo["languages_url"], headers=headers).json()
            for lang in lang_response:
                repo_languages[lang] = repo_languages.get(lang, 0) + 1

            commits_response = requests.get(repo["url"] + "/commits", headers=headers)
            if commits_response.status_code == 200:
                total_commits += len(commits_response.json())

    return {
        "profile": profile_response,
        "languages": repo_languages,
        "total_commits": total_commits,
    }

### **Summarize GitHub Profile Using LLM** ###
def process_github_data(github_data):
    """Uses LLM to summarize useful GitHub insights."""
    github_prompt = f"""
    Here is GitHub data:

    {json.dumps(github_data, indent=2)}

    Summarize the most relevant programming insights from this user.
    """
    return llm.invoke(github_prompt)

### **Main Function: Find Useful Info** ###
def find_useful_info(user_query):
    """Finds and structures useful information about a hacker/programmer."""
    record_id = extract_record_id(user_query)
    if not record_id:
        return "No valid Airtable record ID found in your query."

    fields, table_name = search_airtable_record(record_id, user_query)
    if not fields:
        return "Record not found in Airtable."

    useful_info = extract_useful_fields(fields, user_query)

    # Fetch GitHub Data if available
    github_summary = None
    if "GitHub" in useful_info:
        github_data = fetch_github_data(useful_info["GitHub"])
        github_summary = process_github_data(github_data)

    response = {
        "Table": table_name,
        "Airtable Info": useful_info,
        "GitHub Summary": github_summary,
    }

    return json.dumps(response, indent=2)

### **LangChain Tools** ###
tools = [
    Tool(
        name="Find Hacker Info",
        func=lambda params: find_useful_info(params["query"]),
        description="Searches Airtable for a hacker's information and fetches GitHub data.",
    )
]

# Load Chat Prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Iterate through the prompt to replace `{input}` with `{query}` where necessary
for message in prompt:
    if isinstance(message, SystemMessage) or isinstance(message, HumanMessage):
        message.content = message.content.replace('{input}', '{query}')

# Create conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create structured chat agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# Create AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# Chat Loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    memory.chat_memory.add_message(HumanMessage(content=user_input))
    response = agent_executor.invoke({"input": user_input, "chat_history": memory.chat_memory.get_messages()})
    print("Bot:", response["output"])
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
