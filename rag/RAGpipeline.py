import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
# ----------------- Step 0: Prepare data -----------------
import dotenv
dotenv.load_dotenv()

CHROMA_PATH = "chroma_db"

FOLDER_PATH='Chinese-Laws'

# ----------------- Step 3: Embedding and storage -----------------
# Use OpenAIEmbeddings to create embedding model
print("3. Creating embeddings and storing to vector database...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Use Chroma as vector store
# Default storage in local .chroma_db folder
if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):

    print(f"🆕 Database not detected, creating and storing to {CHROMA_PATH}...")
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    # ----------------- Step 1: Load documents -----------------
    print("1. Loading documents...")
    loader = DirectoryLoader(
    path=FOLDER_PATH,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}  
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    # ----------------- Step 2: Split documents -----------------
    print("2. Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # Maximum length of each chunk
        chunk_overlap=50, # Overlap length between chunks
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(splits)} chunks")
    # Execute time-consuming embedding operations here
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    print("✅ Database creation completed!")
else:
    print(f"🔄 Detected existing vector database ({CHROMA_PATH}), loading directly...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
# Set vector store as retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ----------------- Step 4 & 5: Define RAG chain and query -----------------
# 4. Define prompt template for enhanced generation
prompt = ChatPromptTemplate.from_template("""
你是一个专业的法律助手。请基于以下提供的【法律法规上下文】来回答用户的问题。

要求：
1. 回答必须**基于提供的上下文**，不要编造法律条文。
2. 如果上下文不足以回答问题，请直接说“根据现有资料无法回答”。

【法律法规上下文】:
{context}

【用户问题】:
{input}

【回答】:
""")






# ----------------- Run query -----------------
query = "走私罪怎么判刑？"

print(f"🔍 Retrieving: {query}")
retrieved_docs = retriever.invoke(query)
# Print results
print(f"\n✅ Retrieved {len(retrieved_docs)} relevant document(s):\n")

# Combine documents into a single string
context_string = ""
for i, doc in enumerate(retrieved_docs):
    # Get filtered metadata (only crime_small and crime_big)
    
    
    # Build document string
    doc_string = f"--- Document {i+1} ---\n"
    doc_string += f"[Content]: {doc.page_content[:300]}\n"
    
    context_string += doc_string

print(context_string)