# NLP-Context-Engineering

COMS 4705 Project: From Static Retrieval to Dynamic Context: Investigating Context Engineering in RAG-Based QA

## RAG Pipeline Overview

The system processes Chinese legal crime data from JSON Lines format, creates vector embeddings, stores them in a Chroma vector database, and enables semantic search to retrieve relevant legal documents based on user queries.

### Features

- **Document Loading**: Reads legal documents from JSON Lines (JSONL) format
- **Text Chunking**: Splits documents into manageable chunks with overlap for better context retention
- **Vector Embeddings**: Uses OpenAI's `text-embedding-3-small` model for semantic embeddings
- **Vector Database**: Stores and retrieves documents using Chroma vector database with persistence
- **Semantic Search**: Retrieves top-k most relevant documents based on query similarity

### RAG Architecture

The RAG pipeline consists of the following steps:

#### Step 0: Data Preparation
- Loads environment variables from `.env` file
- Imports the document loading function

#### Step 1: Document Loading
- Reads legal documents from `data/kg_crime.json` (JSONL format)
- Converts each JSON object into LangChain `Document` objects
- Extracts metadata: crime links, crime categories (big/small)
- Processes legal fields: `fatiao`, `gainian`, `tezheng`, `chufa`, `rending`, `jieshi`, `bianhu`

#### Step 2: Document Splitting
- Uses `RecursiveCharacterTextSplitter` to split documents into chunks
- **Chunk size**: 500 characters
- **Chunk overlap**: 50 characters (for context continuity)

#### Step 3: Embedding and Storage
- Creates embeddings using OpenAI's embedding model
- Stores embeddings in Chroma vector database
- **Smart loading**: If database exists, loads directly (fast); otherwise creates new database
- Database persists to `./chroma_db_legal` directory

#### Step 4: Retrieval Setup
- Configures retriever with `k=10` (retrieves top 10 most relevant documents)
- Uses semantic similarity search

#### Step 5: Query Execution
- Performs semantic search on user queries
- Returns relevant documents with metadata
- Displays document summaries and metadata

## Requirements

### Python Packages
```
langchain
langchain-openai
langchain-community
langchain-text-splitters
chromadb
python-dotenv
```



## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd NLP-Context-Engineering
```

2. Install dependencies:
```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters chromadb python-dotenv
```

3. Set up environment variables:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```




## Example Output

```
1. Loading JSON Lines data from file data/kg_crime.json...
   Successfully loaded 857 legal document(s).
2. Splitting documents...
3. Creating embeddings and storing to vector database...
🔄 Detected existing vector database (./chroma_db_legal), loading directly...
🔍 Retrieving: 走私罪怎么判刑？

✅ Retrieved 10 relevant document(s):

--- Document 1 ---
[Content Summary]: --- fatiao ---
[刑法条文]
第一百五十一条走私武器、弹药、核材料或者伪造的货币的，处七年以上有期徒刑...
[Metadata]: {'source': 'http://china.findlaw.cn/zuiming/...', 'crime_big': '破坏社会主义市场经济秩序罪', 'crime_small': '走私罪'}
```

## Configuration

### Adjusting Chunk Parameters

In `RAGpipeline.py`, modify the `RecursiveCharacterTextSplitter` parameters:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Adjust chunk size
    chunk_overlap=50,    # Adjust overlap
    length_function=len
)
```

### Changing Retrieval Count

Modify the `k` parameter in the retriever:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Change 10 to desired number
```

## Data Format

The input JSONL file should have the following structure (one JSON object per line):

```json
{
  "_id": {...},
  "crime_big": "危害国家安全罪",
  "crime_small": "背叛国家罪",
  "crime_link": "http://...",
  "fatiao": [...],
  "gainian": [...],
  "tezheng": [...],
  "chufa": [...],
  "rending": [...],
  "jieshi": [...],
  "bianhu": [...]
}
```

## Notes

- The vector database is persisted locally, so subsequent runs will be much faster
- To rebuild the database, delete the `chroma_db_legal` directory

- All retrieved documents include metadata for traceability

## License

This project is part of COMS 4705 coursework at Columbia University.
