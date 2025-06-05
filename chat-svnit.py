from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vectordb = FAISS.load_local("embeddings_db", embeddings=embedding, allow_dangerous_deserialization=True)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 25})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

question = input("Enter your question: ")

docs = retriever.invoke(question)
context = "\n\n".join([doc.page_content for doc in docs])

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful chatbot assistant with expertise in extracting and answering questions from academic documents."
     "Use the provided context to answer the user's query."
     "If the answer is not found, respond with 'I don't know.' And you may suggest the sources where user can find the information, Do not hallucinate."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])


parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({
    "context": context,
    "question": question
})


print(result)