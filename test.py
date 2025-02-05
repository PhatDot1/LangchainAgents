import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent


load_dotenv()

OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")





# Define a very simple tool function that returns the current time 
def get_current_time(*args, **kwargs):
"""Returns the current time in H:MM AM/PM format."""
    import datetime # Import datetime module to get current time
    now = datetime.datetime.now() # Get current time
    return now.strftime("%I:%M %p") # Format time in H:MM AM/PM format

    
# List of tools available to the agent
tools = [
    Tool (
        name="Time", # Name of the tool
        func=get_current_time, # Function that the tool will execute 
        description="Useful for when you need to know the current time",
    ),
]


















def main():
    llm = OpenAI(openai_api_key=OPEN_AI_API_KEY)
    result = llm.predict("list me three fruits")
    print(result)








if __name__ == "__main__":
    main()