from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Initialize embedding model
embedding = OpenAIEmbeddings()

# Load FAISS vector store
vectordb = FAISS.load_local("embeddings_db", embeddings=embedding, allow_dangerous_deserialization=True)

# Create retriever
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 25})

# Chat LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.01)

# Memory for conversation
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Prompt template with memory placeholder
prompt = ChatPromptTemplate([
    ("system","Your name is syntra and You are a helpful chatbot assistant with expertise in extracting and answering questions of SVNIT, Surat."
        "Use the provided context and chat history to answer the user's query."
        "If the answer is not found, respond with 'I don't know.' You may also suggest source of information if informtaion is not available in context. Do not hallucinate."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human"," Context:\n{context}\n\nQuestion: {question}"),
])

# Chain
parser = StrOutputParser()
chain = prompt | llm | parser

# REPL loop with memory
print("🔁 Hello I'm Syntra, Your helpful chatbot assistant for SVNIT, Surat. Type 'exit' to stop.\n")

while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    result = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": memory.chat_memory.messages  # Include chat history in the prompt
    })

    # Update memory
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(result)

    print(f"Syntra: {result}\n")