import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# Set page config
st.set_page_config(page_title="Syntra - SVNIT Chatbot", page_icon="🤖")

# Title
st.title("🤖 Syntra - SVNIT Chatbot")
st.markdown("###### Hello I'm Syntra, Your helpful chatbot assistant for Computer Science Engineering Department of SVNIT, Surat.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load vector store and LLM only once
@st.cache_resource(show_spinner="Loading models...")
def load_chain():
    embedding = OpenAIEmbeddings()
    vectordb = FAISS.load_local("embeddings_db", embeddings=embedding, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 25})

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.01)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Your name is Syntra and you are a helpful chatbot assistant with expertise in extracting and answering questions of SVNIT, Surat. "
                   "Use the provided context and chat history to answer the user's query. "
                   "If the answer is not found, respond with 'I don't know.' You may also suggest a source of information if it is not available in context. Do not hallucinate."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    parser = StrOutputParser()
    chain = prompt | llm | parser

    return retriever, chain

retriever, chain = load_chain()

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "ai"):
        st.markdown(msg.content)

# User input
user_input = st.chat_input("Ask me something about SVNIT...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)

    # Retrieve context
    docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Run the chain
    response = chain.invoke({
        "context": context,
        "question": user_input,
        "chat_history": st.session_state.chat_history
    })

    # Display AI response
    st.chat_message("ai").markdown(response)

    # Update session state
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
