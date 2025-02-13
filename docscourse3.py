#!/usr/bin/env python
"""
Full code for a course content generator tool.

This script:
 - Recursively scrapes documentation pages from one or more base URLs.
 - Summarizes each page to reduce content size.
 - Batches the summaries to remain under token limits.
 - Uses an LLM (GPT-4 via LangChain) to generate structured course lessons.
 - Saves each lesson as a text file in your current working directory.
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

def generate_course_lessons(all_pages):
    """
    Generates structured course lessons from the summarized page content.
    The output is expected to follow a format where each lesson starts with a header (e.g., "Lesson1: ...")
    followed by bullet points (lines starting with "-") that list URLs.
    """
    summaries = summarize_each_page(all_pages)
    batches = dynamic_batching(summaries)
    lessons = {}

    for batch in batches:
        batch_summaries = {}
        for url_tuple in batch:
            url = url_tuple[0]
            if url in summaries:
                batch_summaries[url] = summaries[url]
            else:
                print(f"⚠ Warning: URL {url} is missing from summaries. Skipping...")
                continue

        if not batch_summaries:
            continue

        content_summary = "\n\n".join(
            [f"Page: {url}\nSummary: {summary}" for url, summary in batch_summaries.items()]
        )

        # Instruct the LLM to output lessons with bullet points starting with '-' for each URL.
        prompt = f"""
You are an AI that creates structured educational courses. Given the following summarized documentation:
{content_summary}

Please generate structured lessons, logically dividing topics into separate lessons.
Format the output as follows:
Lesson1: [Optional title or description]
- https://URL1
- https://URL2

Lesson2: [Optional title or description]
- https://URL3
- https://URL4
"""
        lesson_mapping = llm.invoke(prompt).content
        print(f"📝 Raw LLM Output:\n{lesson_mapping}")

        # Parse the output manually line-by-line.
        extracted_lessons = {}
        current_lesson = None
        for line in lesson_mapping.splitlines():
            line = line.strip()
            if line.lower().startswith("lesson"):
                # Create a new lesson identifier (e.g., "Lesson1")
                parts = line.split(":", 1)
                lesson_id = parts[0].replace(" ", "")
                current_lesson = lesson_id
                extracted_lessons[current_lesson] = []
            elif line.startswith("-") and current_lesson is not None:
                url_candidate = line.lstrip("-").strip()
                # Check if the candidate looks like a URL and is present in all_pages.
                if url_candidate.startswith("http") and url_candidate in all_pages:
                    extracted_lessons[current_lesson].append(url_candidate)
        # Merge the extracted lessons into our lessons dictionary.
        for lesson_name, urls in extracted_lessons.items():
            if lesson_name in lessons:
                lessons[lesson_name].extend(urls)
            else:
                lessons[lesson_name] = urls

    return lessons

def save_lessons(all_pages):
    """
    Saves each lesson as a text file in the current working directory.
    If a lesson's content is too large, it is split into multiple files.
    """
    lessons = generate_course_lessons(all_pages)
    for lesson, urls in lessons.items():
        content = "\n\n".join([f"Page: {url}\n{all_pages.get(url, '')}" for url in urls])
        if count_tokens(content) > SAFE_LIMIT:
            print(f"⚠️ Lesson {lesson} is too large, splitting into multiple pages...")
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

            for i, part_content in enumerate(split_content):
                file_name = f"{lesson}-page{i+1}.txt"
                file_path = os.path.join(os.getcwd(), file_name)
                print(f"✅ Saving {file_path}...")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(part_content)
        else:
            file_name = f"{lesson}.txt"
            file_path = os.path.join(os.getcwd(), file_name)
            print(f"✅ Saving {file_path}...")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

def generate_course_from_links(links_input: str):
    """
    Given a comma-separated list of documentation URLs, this function:
      - Scrapes all pages (recursively) from each base URL.
      - Generates course lessons.
      - Saves the lessons as text files in the current directory.
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
    print("📚 Generating and saving structured lessons...")
    save_lessons(all_pages)
    return f"Course content generated and saved in {os.getcwd()}."

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
