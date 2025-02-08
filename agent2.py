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
    except:
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

# Initialize a ChatOpenAI model with the API key
llm = ChatOpenAI(
    openai_api_key=OPEN_AI_API_KEY,  # Pass the API key directly
    model="gpt-4",  # Ensure correct model name (e.g., 'gpt-4')
)

# Create a structured Chat Agent with Conversation Buffer Memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

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
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat Loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
