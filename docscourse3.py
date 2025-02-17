# switch this one to claude to get around text limits.
"""
Full code for a course content generator tool.

This script:
 - Recursively scrapes documentation pages from one or more base URLs.
 - Summarizes each page to reduce content size.
 - Batches the summaries to remain under token limits.
 - Uses an LLM (GPT-4 via LangChain) to generate structured course lessons.
 - Saves the generated curriculum as text file(s) in your current working directory.
 - Wraps all of the above as a LangChain Tool so that an agent can be created,
   and you can simply type queries like:
      "Give me a course content covering: https://docs.anchorprotocol.com, https://docs.solana.com"
   and the tool will do its work.
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import tiktoken
from dotenv import load_dotenv

# LangChain and agent imports
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ------------------------
# Environment & LLM Setup
# ------------------------

load_dotenv()
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")

# Initialize the ChatOpenAI model (ensure your API key is loaded)
llm = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, model="gpt-4")

# Set token limits
TOKEN_LIMIT = 8192
SAFE_LIMIT = int(TOKEN_LIMIT * 0.8)  # 80% of the max token limit as a buffer

# ------------------------
# Helper Functions
# ------------------------

def fetch_page_content(url):
    """Fetch the content of a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        return f"Failed to fetch {url}: {str(e)}"

def get_all_links(base_url, visited=None):
    """
    Recursively collect all internal links from the given base URL.
    (Only collects links on the same domain as the base URL.)
    """
    if visited is None:
        visited = set()
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
            # Only follow links on the same domain
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                get_all_links(full_url, visited)
    except requests.RequestException:
        pass
    return visited

def count_tokens(text):
    """Counts the number of tokens in the given text using tiktoken for gpt-4."""
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))

def summarize_each_page(all_pages):
    """
    Summarizes each scraped page to reduce the input size.
    If a page is too long, it splits the content into parts.
    """
    summaries = {}
    for url, content in all_pages.items():
        token_count = count_tokens(content)
        if token_count > SAFE_LIMIT:
            print(f"⚠️ Page {url} is too large ({token_count} tokens). Splitting content...")
            split_content = []
            words = content.split()
            part = []
            for word in words:
                part.append(word)
                if count_tokens(" ".join(part)) >= SAFE_LIMIT:
                    split_content.append(" ".join(part))
                    part = []
            if part:
                split_content.append(" ".join(part))
            summaries[url] = split_content
        else:
            prompt = f"Summarize the following content in 100 words:\n{content[:1000]}"
            summary = llm.invoke(prompt).content
            summaries[url] = summary
    return summaries

def dynamic_batching(summaries):
    """
    Dynamically groups page summaries into batches such that each batch stays under token limits.
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for url, summary in summaries.items():
        if isinstance(summary, list):  # if the page was split into parts
            for i, part in enumerate(summary):
                token_count = count_tokens(part)
                if current_tokens + token_count > SAFE_LIMIT:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                current_batch.append((url, f"{url}-part{i+1}", part))
                current_tokens += token_count
        else:
            token_count = count_tokens(summary)
            if current_tokens + token_count > SAFE_LIMIT:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append((url, url, summary))
            current_tokens += token_count

    if current_batch:
        batches.append(current_batch)
    return batches

# ------------------------
# New Curriculum Synthesis Functions
# ------------------------

def dynamic_batching_entries(all_pages, summaries):
    """
    Creates entries combining a truncated raw excerpt (first 500 characters)
    and the summary for each URL, and then batches these entries such that each batch is under token limits.
    """
    entries = []
    for url in summaries:
        summary = summaries[url]
        summary_text = " ".join(summary) if isinstance(summary, list) else summary
        raw_content = all_pages.get(url, "")
        raw_excerpt = raw_content[:500]  # truncate to 500 characters to save tokens
        entry = f"URL: {url}\nRaw Excerpt: {raw_excerpt}\nSummary: {summary_text}"
        entries.append(entry)
    batches = []
    current_batch = []
    current_tokens = 0
    for entry in entries:
        entry_tokens = count_tokens(entry)
        if current_tokens + entry_tokens > SAFE_LIMIT:
            batches.append("\n\n".join(current_batch))
            current_batch = [entry]
            current_tokens = entry_tokens
        else:
            current_batch.append(entry)
            current_tokens += entry_tokens
    if current_batch:
        batches.append("\n\n".join(current_batch))
    return batches

def generate_course_curriculum(all_pages):
    """
    Generates a comprehensive course curriculum by synthesizing both the raw page content and the summaries.
    The curriculum is divided into chapters/lessons. For each chapter, include:
      - A clear title
      - An introduction outlining key concepts and learning objectives
      - Detailed lesson content that accurately explains technical details (quoting raw content as needed)
      - A summary of key takeaways
      - Optional further resources or suggested readings
    """
    summaries = summarize_each_page(all_pages)
    entry_batches = dynamic_batching_entries(all_pages, summaries)
    curriculum_parts = []
    
    for batch_info in entry_batches:
        prompt = f"""
You are an expert educator and curriculum designer. Using the following source information from documentation (which includes both raw excerpts and summaries), synthesize a comprehensive and original course curriculum. The curriculum should be divided into chapters or lessons. For each chapter, include:
- A clear title
- An introduction outlining key concepts and learning objectives
- Detailed lesson content that accurately explains technical details (quoting raw content as needed)
- A summary of key takeaways
- Optional further resources or suggested readings

Source Material:
{batch_info}

Course Curriculum:
"""
        curriculum_part = llm.invoke(prompt).content
        curriculum_parts.append(curriculum_part)
    
    final_curriculum = "\n\n".join(curriculum_parts)
    return final_curriculum

def save_curriculum(all_pages):
    """
    Generates the full course curriculum and saves it as text file(s) in the current working directory.
    If the curriculum is too large, it is split into multiple files.
    """
    curriculum = generate_course_curriculum(all_pages)
    if count_tokens(curriculum) > SAFE_LIMIT:
        print("⚠️ Curriculum is too large, splitting into multiple files...")
        split_content = []
        words = curriculum.split()
        part = []
        for word in words:
            part.append(word)
            if count_tokens(" ".join(part)) >= SAFE_LIMIT:
                split_content.append(" ".join(part))
                part = []
        if part:
            split_content.append(" ".join(part))
        for i, part_content in enumerate(split_content):
            file_name = f"Course_Curriculum_part{i+1}.txt"
            file_path = os.path.join(os.getcwd(), file_name)
            print(f"✅ Saving {file_path}...")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(part_content)
    else:
        file_name = "Course_Curriculum.txt"
        file_path = os.path.join(os.getcwd(), file_name)
        print(f"✅ Saving {file_path}...")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(curriculum)

# ------------------------
# Updated Course Generation Function
# ------------------------

def generate_course_from_links(links_input: str):
    """
    Given a comma-separated list of documentation URLs, this function:
      - Scrapes all pages (recursively) from each base URL.
      - Synthesizes a comprehensive course curriculum using both raw page content and summaries.
      - Saves the curriculum as text file(s) in the current directory.
    Returns a message indicating where the files were saved.
    """
    urls = [link.strip() for link in links_input.split(",") if link.strip()]
    all_pages = {}
    for url in urls:
        print(f"🔎 Fetching all links from {url} ...")
        all_links = get_all_links(url)
        print(f"📄 Found {len(all_links)} pages in {url}.")
        for link in all_links:
            print(f"📝 Scraping: {link}")
            all_pages[link] = fetch_page_content(link)
    print("📚 Generating and saving the course curriculum...")
    save_curriculum(all_pages)
    return f"Course curriculum generated and saved in {os.getcwd()}."

# ------------------------
# LangChain Tool & Agent Setup
# ------------------------

# Wrap our functionality as a LangChain Tool.
course_content_tool = Tool(
    name="CourseContentGenerator",
    func=lambda input, **kwargs: generate_course_from_links(input),
    description="Generates course content from provided documentation URLs. Input should be a comma-separated list of URLs."
)

# Load a structured chat prompt from the LangChain hub.
prompt = hub.pull("hwchase17/structured-chat-agent")

# Create conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a structured chat agent with our tool.
agent = create_structured_chat_agent(llm=llm, tools=[course_content_tool], prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=[course_content_tool], verbose=True, memory=memory, handle_parsing_errors=True
)

# Set an initial system message.
initial_message = (
    "You are an AI assistant that can generate educational course content by scraping documentation links. "
    "When a user asks for course content (e.g., 'Give me a course content covering: <comma-separated URLs>'), "
    "you should use the CourseContentGenerator tool."
)
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# ------------------------
# Main Interaction Loop
# ------------------------

if __name__ == "__main__":
    print("Welcome to the Course Content Generator!")
    print("Type your query (e.g., 'Give me a course content covering: https://docs.anchorprotocol.com, https://docs.solana.com') or 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        response = agent_executor.invoke({"input": user_input})
        print("Bot:", response["output"])
