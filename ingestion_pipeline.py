import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import glob
load_dotenv()
from pypdf import PdfReader


def extract_pdf(file):
    text = ''

    reader = PdfReader(file)
    for page in   reader.pages:

      
       text = text + page.extract_text()
    return text



def text_splitter(doc,chunks=600,chunk_overlap=2):
    splitter = RecursiveCharacterTextSplitter(
        chunk=chunks,
        chunk_overlap= chunk_overlap
    )
    return splitter.split_text(doc)


def vector_store(persist_directory = './database_dir'): # enter whatever persist directory you need

    vector_model = OpenAIEmbeddings(vector_model='text-embedding-3-small')
    create_vector_store = Chroma(
       collection_name= 'embeddings',
       persist_directory = persist_directory,
       collection_metadata='hn'


    )




def integrate ():

    text =extract_pdf('./psychology_of_money.pdf')#write your own file name
    chunks = text_splitter(text)
    vector_d = vector_store ()