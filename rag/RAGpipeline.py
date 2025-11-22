import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
# ----------------- Step 0: Prepare data -----------------
from read_rag_data import load_json_to_documents
import dotenv
dotenv.load_dotenv()
# ----------------- Step 1: Load documents -----------------
documents = load_json_to_documents()
CHROMA_PATH = "./chroma_db_legal"
# ----------------- Step 2: Split documents -----------------
# Use RecursiveCharacterTextSplitter to split documents into chunks
print("2. Splitting documents...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # Maximum length of each chunk
    chunk_overlap=50, # Overlap length between chunks
    length_function=len
)
splits = text_splitter.split_documents(documents)

# ----------------- Step 3: Embedding and storage -----------------
# Use OpenAIEmbeddings to create embedding model
print("3. Creating embeddings and storing to vector database...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Use Chroma as vector store
# Default storage in local .chroma_db folder
if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
    print(f"🔄 Detected existing vector database ({CHROMA_PATH}), loading directly...")
    
    # Directly load existing database without embedding computation, very fast
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
else:
    print(f"🆕 Database not detected, creating and storing to {CHROMA_PATH}...")
    
    # Execute time-consuming embedding operations here
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    print("✅ Database creation completed!")

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
for i, doc in enumerate(retrieved_docs):
    print(f"--- Document {i+1} ---")
    # Get content
    print(f"[Content Summary]: {doc.page_content[:100]}...") 
    # Get metadata (crime name, link, etc.)
    print(f"[Metadata]: {doc.metadata}")
    print("\n")