import json
import os
from langchain_core.documents import Document
from typing import List
filename = "data/kg_crime.json"

def load_json_to_documents(file_path: str=filename) -> List[Document]:
    print(f"1. Loading JSON Lines data from file {file_path}...")
    
    data = []
    # --- Start: Read line by line ---
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue # Skip bad data
    # --- End ---

    documents: List[Document] = []
    
    # Process the loaded data list
    for item in data:
        metadata = {
            "source": item.get("crime_link", "N/A"),
            "crime_big": item.get("crime_big", "N/A"),
            "crime_small": item.get("crime_small", "N/A"),
        }

        content_parts = []
        # Process field logic
        for field in ["fatiao", "gainian", "tezheng", "chufa", "rending", "jieshi", "bianhu"]:
            if field in item and item[field]:
                if isinstance(item[field], list):
                    content_parts.append(f"--- {field} ---")
                    content_parts.extend(item[field])
                elif isinstance(item[field], str):
                    content_parts.append(f"--- {field} ---")
                    content_parts.append(item[field])

        full_content = "\n".join(content_parts)
        
        if full_content.strip():
            doc = Document(page_content=full_content, metadata=metadata)
            documents.append(doc)

    print(f"   Successfully loaded {len(documents)} legal document(s).")
    return documents
documents = load_json_to_documents(filename)

