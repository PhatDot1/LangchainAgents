from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Load environment variables
load_dotenv()
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    openai_api_key=OPEN_AI_API_KEY,
    model="gpt-4",
)

# Define Tools
def fetch_page_content(url):
    """Fetches the content of a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        return f"Failed to fetch {url}: {str(e)}"

def get_all_links(base_url, visited=set()):
    """Recursively collects all links from the documentation."""
    if base_url in visited:
        return visited
    
    visited.add(base_url)
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                get_all_links(full_url, visited)
    except requests.RequestException:
        pass
    
    return visited

def generate_course_transcript(all_pages):
    """Generates a course transcript using LLM."""
    content_summary = "\n\n".join([f"Page: {url}\nContent: {content[:1000]}..." for url, content in all_pages.items()])
    prompt = f"""
    You are an AI that creates structured educational courses. Given the following documentation:
    {content_summary}
    
    Please generate a well-structured course transcript covering all aspects of this ecosystem, with clear sections and explanations.
    """
    
    return llm.invoke(prompt).content

# Define the tools
tools = [
    Tool(
        name="FetchPageContent",
        func=fetch_page_content,
        description="Fetches the text content of a given URL.",
    ),
    Tool(
        name="GetAllLinks",
        func=lambda url: list(get_all_links(url)),
        description="Recursively collects all links from a documentation base URL.",
    ),
    Tool(
        name="GenerateCourseTranscript",
        func=generate_course_transcript,
        description="Generates a structured course transcript from scraped documentation.",
    ),
]

# Load the correct JSON Chat Prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Create a structured Chat Agent with Conversation Buffer Memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# Create the structured chat agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# Create an AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# Initial system message
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: FetchPageContent, GetAllLinks, and GenerateCourseTranscript."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat Loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    
    memory.chat_memory.add_message(HumanMessage(content=user_input))
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])
    memory.chat_memory.add_message(AIMessage(content=response["output"]))