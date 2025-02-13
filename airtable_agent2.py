import os
import json
import requests
from dotenv import load_dotenv

# Import LangChain components.
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# =============================================================================
# Load Environment Variables
# =============================================================================

load_dotenv()

# Your OpenAI API key (for the LLM powering the agent)
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
# Your Airtable API key and (optionally) the client secret.
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_CLIENT_SECRET = os.getenv("AIRTABLE_CLIENT_SECRET")  # May be None if not required

# Base URLs for the standard API and the Metadata API.
BASE_URL = "https://api.airtable.com/v0"
META_URL = "https://api.airtable.com/v0/meta"

# =============================================================================
# Define Airtable API Tool Functions
# =============================================================================

def list_bases() -> str:
    """List all Airtable bases accessible with the API key."""
    url = f"{META_URL}/bases"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    if AIRTABLE_CLIENT_SECRET:
        headers["X-Airtable-Client-Secret"] = AIRTABLE_CLIENT_SECRET
    response = requests.get(url, headers=headers)
    if response.ok:
        data = response.json()
        bases = data.get("bases", [])
        if not bases:
            return "No bases found."
        # Format a simple list of base IDs and names.
        bases_info = "\n".join([f"{base['id']}: {base['name']}" for base in bases])
        return f"List of Bases:\n{bases_info}"
    else:
        return f"Error listing bases: {response.status_code} {response.text}"

def get_base_schema(base_id: str) -> str:
    """Retrieve the schema (tables and fields) for a given base."""
    url = f"{META_URL}/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    if AIRTABLE_CLIENT_SECRET:
        headers["X-Airtable-Client-Secret"] = AIRTABLE_CLIENT_SECRET
    response = requests.get(url, headers=headers)
    if response.ok:
        data = response.json()
        tables = data.get("tables", [])
        if not tables:
            return f"No tables found in base {base_id}."
        # List table IDs and names.
        tables_info = "\n".join([f"{table['id']}: {table['name']}" for table in tables])
        return f"Schema for Base {base_id}:\n{tables_info}"
    else:
        return f"Error retrieving schema: {response.status_code} {response.text}"

def list_records(args: str) -> str:
    """
    List records from a specific table.
    
    Expected input format (JSON string or comma‐separated values):
      {
        "base_id": "yourBaseId",
        "table_name": "yourTableName"
      }
    """
    try:
        # Try to parse a JSON input.
        params = json.loads(args)
    except json.JSONDecodeError:
        # Alternatively, assume comma-separated input.
        parts = [s.strip() for s in args.split(",")]
        if len(parts) < 2:
            return "Please provide both base_id and table_name."
        params = {"base_id": parts[0], "table_name": parts[1]}
    
    base_id = params.get("base_id")
    table_name = params.get("table_name")
    if not base_id or not table_name:
        return "Missing base_id or table_name."

    url = f"{BASE_URL}/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    response = requests.get(url, headers=headers)
    if response.ok:
        data = response.json()
        records = data.get("records", [])
        if not records:
            return f"No records found in table '{table_name}'."
        # Summarize each record.
        recs = "\n".join([f"ID: {record['id']}, Fields: {record.get('fields')}" for record in records])
        return f"Records in {table_name}:\n{recs}"
    else:
        return f"Error listing records: {response.status_code} {response.text}"

def create_record(args: str) -> str:
    """
    Create a record in a specified table.
    
    Expected input (JSON):
      {
        "base_id": "yourBaseId",
        "table_name": "yourTableName",
        "fields": {
          "Field1": "Value1",
          "Field2": "Value2",
          ...
        }
      }
    """
    try:
        params = json.loads(args)
    except Exception as e:
        return f"Error parsing input: {e}"
    
    base_id = params.get("base_id")
    table_name = params.get("table_name")
    fields = params.get("fields")
    if not base_id or not table_name or not fields:
        return "Missing base_id, table_name, or fields."
    
    url = f"{BASE_URL}/{base_id}/{table_name}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"fields": fields}
    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        data = response.json()
        return f"Record created successfully with ID: {data.get('id', 'Unknown')}"
    else:
        return f"Error creating record: {response.status_code} {response.text}"

def update_record(args: str) -> str:
    """
    Update a record in a table.
    
    Expected input (JSON):
      {
        "base_id": "yourBaseId",
        "table_name": "yourTableName",
        "record_id": "recordIdToUpdate",
        "fields": {
          "Field1": "NewValue",
          ...
        }
      }
    """
    try:
        params = json.loads(args)
    except Exception as e:
        return f"Error parsing input: {e}"
    
    base_id = params.get("base_id")
    table_name = params.get("table_name")
    record_id = params.get("record_id")
    fields = params.get("fields")
    if not base_id or not table_name or not record_id or not fields:
        return "Missing base_id, table_name, record_id, or fields."
    
    url = f"{BASE_URL}/{base_id}/{table_name}/{record_id}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"fields": fields}
    response = requests.patch(url, headers=headers, json=payload)
    if response.ok:
        data = response.json()
        return f"Record updated successfully. ID: {data.get('id', 'Unknown')}"
    else:
        return f"Error updating record: {response.status_code} {response.text}"

def delete_record(args: str) -> str:
    """
    Delete a record from a table.
    
    Expected input (JSON):
      {
        "base_id": "yourBaseId",
        "table_name": "yourTableName",
        "record_id": "recordIdToDelete"
      }
    """
    try:
        params = json.loads(args)
    except Exception as e:
        return f"Error parsing input: {e}"
    
    base_id = params.get("base_id")
    table_name = params.get("table_name")
    record_id = params.get("record_id")
    if not base_id or not table_name or not record_id:
        return "Missing base_id, table_name, or record_id."
    
    url = f"{BASE_URL}/{base_id}/{table_name}/{record_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    response = requests.delete(url, headers=headers)
    if response.ok:
        return f"Record {record_id} deleted successfully."
    else:
        return f"Error deleting record: {response.status_code} {response.text}"

# =============================================================================
# Configure Agent Tools
# =============================================================================

tools = [
    Tool(
        name="List Bases",
        func=list_bases,
        description="Lists all Airtable bases accessible with your API key."
    ),
    Tool(
        name="Get Base Schema",
        func=get_base_schema,
        description="Retrieves the schema (tables and fields) for a specific base. Input: base_id."
    ),
    Tool(
        name="List Records",
        func=list_records,
        description=("Lists records in a table. Input should provide base_id and table_name "
                     "in JSON format, for example: "
                     '{"base_id": "app123", "table_name": "MyTable"}.')
    ),
    Tool(
        name="Create Record",
        func=create_record,
        description=("Creates a record in a table. Input must include base_id, table_name, and a JSON "
                     "object of fields, for example: "
                     '{"base_id": "app123", "table_name": "MyTable", "fields": {"Name": "John", "Age": 30}}.')
    ),
    Tool(
        name="Update Record",
        func=update_record,
        description=("Updates a record in a table. Input must include base_id, table_name, record_id, "
                     "and the fields to update (as JSON).")
    ),
    Tool(
        name="Delete Record",
        func=delete_record,
        description=("Deletes a record from a table. Input must include base_id, table_name, and record_id "
                     "in JSON format.")
    ),
    # (Add additional tools for other Airtable endpoints as needed.)
]

# =============================================================================
# Set Up the Agent
# =============================================================================

# Pull a structured chat prompt from the Hub (or define your own prompt)
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize the ChatOpenAI model (using GPT-4 in this example)
llm = ChatOpenAI(
    openai_api_key=OPEN_AI_API_KEY,
    model="gpt-4"
)

# Set up conversation memory so the agent can track context over the chat
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a structured chat agent that can dynamically select the Airtable tools.
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# Wrap the agent and tools in an AgentExecutor to handle interaction.
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# Add an initial system message to instruct the agent.
initial_message = (
    "You are an AI assistant for Airtable. Your job is to help users interact with Airtable's API. "
    "You can perform tasks like listing bases, retrieving a base's schema, listing, creating, updating, and deleting records. "
    "If details such as the base_id, table name, or fields are not provided, ask clarifying questions."
)
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# =============================================================================
# Chat Loop
# =============================================================================

print("Welcome to the Airtable Agent. Type 'exit' to quit.")
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add the user's message to conversation memory.
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user's input.
    result = agent_executor.invoke({"input": user_input})
    output = result.get("output", "")
    print("Bot:", output)

    # Store the agent's response in memory.
    memory.chat_memory.add_message(AIMessage(content=output))
