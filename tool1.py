import os
from dotenv import load_dotenv  # Load environment variables from .env file
from langchain import hub  # For pulling prompts from the hub
from langchain.agents import AgentExecutor, create_tool_calling_agent  # For creating and executing agents
from langchain.chains import create_history_aware_retriever, create_retrieval_chain  # For creating chains
from langchain.chains.combine_documents import create_stuff_documents_chain  # For combining documents
from langchain_community.vectorstores import Chroma  # For vector store support
from langchain_core.messages import AIMessage, HumanMessage  # For message types in the agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # For message placeholders in prompts
from langchain_core.tools import Tool  # To define tools used by the agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # For interacting with OpenAI models
import wikipedia  # For searching and summarizing Wikipedia
import datetime  # For working with date and time
from pydantic import BaseModel, Field  # Now importing directly from pydantic

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if not OPEN_AI_API_KEY:
    print("API Key not found. Please check your .env file.")
else:
    print("API Key successfully loaded.")

# Load the existing Chroma vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "..", "..", "4_rag", "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Check if the Chroma vector store already exists
if os.path.exists(persistent_directory):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=None)
else:
    raise FileNotFoundError(f"The directory {persistent_directory} does not exist. Please check the path.")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatOpenAI model
llm = ChatOpenAI(
    openai_api_key=OPEN_AI_API_KEY,  # Pass the API key directly
    model="gpt-4",  # Ensure the correct model name (e.g., 'gpt-4')
)

# Contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
)

# Set Up ReAct Agent with Document Store Retriever
react_docstore_prompt = hub.pull("hwchase17/react")

tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="useful for when you need to answer questions about the context",
    )
]

# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,
)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke(
        {"input": query, "chat_history": chat_history}
    )
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))
