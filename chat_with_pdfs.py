import platform
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from config import OPENAI_API_KEY

class bcolors:
    GREEN = '\033[92m'
    ENDCOLOR = '\033[0m' 

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Load, convert to text and split pdf file into pages

# pdf_name = "Tesla_Annual_Report_2023_Jan31.pdf"
pdf_name = "Pizzeria Da Nino am Eigerplatz Bern » 031 371 11 31.pdf"

loader = PyPDFLoader(pdf_name)
pages = loader.load_and_split()

# Chunk each page into sections
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len)
sections = text_splitter.split_documents(pages)

# Create FAISS index
faiss_index = FAISS.from_documents(sections, OpenAIEmbeddings())

# Define chain
retriever = faiss_index.as_retriever()
memory = ConversationBufferMemory(
    memory_key='chat_history', 
    return_messages=True, 
    output_key='answer')

chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(),
    retriever=retriever,
    memory=memory,
    verbose=True
)


# Collect user input and simulate chat with pdf
if platform.system() == "Windows":
    eof_key = "<Ctrl+Z>"
else:
    eof_key = "<Ctrl+D>"

print(f'Lets talk with the menu of Pizzeria Da Nino in Bern. What would you like to know? Or press {eof_key} to exit.')
while True:
    try:
        user_input = input('Q:')
        print(f"{bcolors.GREEN} A: {chain({'question': user_input})['answer'].strip()}{bcolors.ENDCOLOR}")
    except EOFError:
        break
    except KeyboardInterrupt:
        break

print("Done")


# Test 1:
#print(chain({'question': 'Which is the most used pizza topping?'}))

# Test 2 (including memory):
#print(chain({'question': 'Which pizza has the most toppings?'}))
#print(chain({'question': 'How many toppings does it have?'}))