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
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader

# Load environment variables
load_dotenv()
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    openai_api_key=OPEN_AI_API_KEY,
    model="gpt-4",
)

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

def summarize_each_page(all_pages):
    """Summarizes each scraped page to reduce input size before creating the course transcript."""
    summaries = {}
    for url, content in all_pages.items():
        prompt = f"Summarize the following content in 100 words:\n{content[:1000]}"
        summary = llm.invoke(prompt).content
        summaries[url] = summary
    return summaries

def generate_course_transcript(all_pages):
    """Creates a structured course transcript using summarized page content."""
    summaries = summarize_each_page(all_pages)
    content_summary = "\n\n".join([f"Page: {url}\nSummary: {summary}" for url, summary in summaries.items()])
    
    prompt = f"""
    You are an AI that creates structured educational courses. Given the following summarized documentation:
    {content_summary}
    
    Please generate a structured course transcript with clear lessons.
    """
    
    return llm.invoke(prompt).content

def save_lessons(all_pages):
    """Saves each page content into lesson files based on logical grouping."""
    transcript = generate_course_transcript(all_pages)
    lesson_mapping = llm.invoke(f"""Organize these lessons into sequential order and determine lesson groupings:
    {transcript}
    Return a structured list in the format:
    Lesson1: [URLs...]
    Lesson2: [URLs...]
    """).content
    
    lessons = {}
    for line in lesson_mapping.split("\n"):
        if line.startswith("Lesson"):
            lesson_name, urls = line.split(":")
            urls = urls.strip().strip("[]").split(", ")
            lessons[lesson_name] = urls
    
    for lesson, urls in lessons.items():
        content = "\n\n".join([f"Page: {url}\n{all_pages.get(url, '')}" for url in urls])
        with open(f"{lesson}.txt", "w", encoding="utf-8") as f:
            f.write(content)

def main():
    base_url = input("Enter the documentation base URL: ")
    
    print("Fetching all documentation links...")
    all_links = get_all_links(base_url)
    print(f"Found {len(all_links)} pages to scrape.")
    
    all_pages = {}
    for link in all_links:
        print(f"Scraping: {link}")
        all_pages[link] = fetch_page_content(link)
    
    print("Generating and saving course transcript...")
    save_lessons(all_pages)
    
    print("All lessons and summaries saved.")

if __name__ == "__main__":
    main()
