from dotenv import load_dotenv
from helper import get_child_urls, save_urls_to_text_file, split_and_embed_urls_documents
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
load_dotenv()

url = "https://www.svnit.ac.in/web/department/computer/"

# html_urls, pdf_urls = get_child_urls(url)


# html_txt_path, pdf_txt_path = save_urls_to_text_file(html_urls, pdf_urls, "svnit_cse_department")

html_txt_path, pdf_txt_path = "html_urls_svnit_cse_department.txt", "pdf_urls_svnit_cse_department.txt"

embedding = OpenAIEmbeddings(
    # model="text-embedding-3-small",
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

split_and_embed_urls_documents(html_txt_path, pdf_txt_path, embedding, text_splitter)

