import json
import re
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
import os

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")

# Define Tools
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary
    try:
        # Limit to two sentences for brevity
        return summary(query, sentences=2)
    except Exception as e:
        return "I couldn't find any information on that."

# Define the tools that the agent can use
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),
]

# Load the correct JSON Chat Prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize a ChatOpenAI model with the API key.
# Note: Use invoke() rather than calling the instance so that you get a text string back.
llm = ChatOpenAI(
    openai_api_key=OPEN_AI_API_KEY,  # Pass the API key directly
    model="gpt-4",  # Ensure correct model name (e.g., 'gpt-4')
)

# Create a structured Chat Agent with Conversation Buffer Memory
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

# Create the structured chat agent using the provided prompt, llm, and tools
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# Create an AgentExecutor to manage the interaction between user input, agent, and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  # Use conversation memory to maintain context
    handle_parsing_errors=True,  # Handle parsing errors gracefully
)

# Initial system message to set the context for the chat
initial_message = (
    "You are an AI assistant that can provide helpful answers using available tools.\n"
    "If you are unable to answer, you can use the following tools: Time and Wikipedia."
)
memory.chat_memory.add_message(SystemMessage(content=initial_message))

def parse_instruction(user_input):
    """
    Use the LLM to parse the user's instruction and return a JSON object with these keys:
      - action: "edit" or "delete"
      - identifier: the record identifier (could be a name, email, record ID, etc.)
      - change_field: if the instruction involves a change (like "change email to" or "change name to"),
                      then this field is the field being changed (e.g. "email" or "name"); otherwise null.
      - new_value: if a change is specified, the new value; otherwise null.
    
    IMPORTANT: The identifier should be the record element that is not being changed.
    For example:
      "Edit the record for Patrick Loughran to change email to Patrick@encode.club"
    should yield an identifier of "Patrick Loughran" (not the new email).
    """
    prompt_text = f"""
Your task is to parse the following instruction into a JSON object with exactly these keys:
- "action": a string that is either "edit" or "delete" (based on the instruction).
- "identifier": a string that identifies the record (this may be a name, an email, a record ID, a hackathon record number, etc.).
- "change_field": if the instruction involves changing a field (e.g. "change email to" or "change name to"),
    then specify which field is being changed (for example "email" or "name"); otherwise, use null.
- "new_value": if a change is specified, the new value; otherwise, use null.

Ensure that the "identifier" is the element that is NOT being changed.
Do not include any additional text in your output; output only valid JSON.

Input: "{user_input}"
"""
    # Use .invoke() so that we can access the content of the returned message.
    response = llm.invoke(prompt_text)
    try:
        parsed = json.loads(response.content.strip())
    except Exception as e:
        print("Error parsing JSON:", e)
        parsed = None
    return parsed

# Chat Loop to interact with the user
while True:
    try:
        user_input = input("User: ")
    except KeyboardInterrupt:
        break

    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Parse the instruction using the LLM
    parsed = parse_instruction(user_input)
    if parsed is None:
        print("Could not parse the instruction. Please try again.")
        continue

    # Extract details from the parsed JSON
    action = parsed.get("action", "unknown")
    identifier = parsed.get("identifier", None)
    change_field = parsed.get("change_field", None)
    new_value = parsed.get("new_value", None)

    # Build an intent description for logging purposes.
    if change_field and new_value:
        intent = (
            f"The user wants to {action} the record identified by '{identifier}' and change "
            f"{change_field} to '{new_value}'."
        )
    else:
        intent = f"The user wants to {action} the record identified by '{identifier}'."

    # Print the extracted identifier and intent.
    print("Extracted Identifier:", identifier)
    print("Intent:", intent)

    # For demonstration, use the identifier for a Wikipedia search.
    if identifier:
        wiki_response = search_wikipedia(identifier)
        print("Wikipedia Search Result:", wiki_response)
        memory.chat_memory.add_message(AIMessage(content=wiki_response))
    else:
        print("No identifier found; skipping Wikipedia search.")
