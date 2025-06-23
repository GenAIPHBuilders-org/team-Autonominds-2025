# vector_store_manager.py

import chromadb
import os
from pathlib import Path
from typing import List, Dict, Any
from embeddings import embed_text

# Let's define a path inside the web_app directory for our persistent storage.
# This keeps our database files organized with the web application.
CHROMA_DATA_PATH = str(Path(__file__).resolve().parent / "web_app" / "chroma_db_store")
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Initialize the ChromaDB client.
# We use PersistentClient to ensure our data is saved to disk.

def build_rag_index(collection_name: str, papers: List[Dict[str, Any]]) -> bool:
    """
    Builds or updates a ChromaDB collection with paper abstracts using SciBERT.
    """
    print(f"INFO: Building RAG index for collection: '{collection_name}' with SciBERT.")
    try:
        # We no longer need to provide an embedding function here, as we provide the embeddings manually.
        collection = client.get_or_create_collection(name=collection_name)

        documents = []
        metadatas = []
        ids = []
        embeddings = [] # We will now generate embeddings ourselves

        for paper in papers:
            abstract = paper.get('abstract')
            doi = paper.get('doi')

            if abstract and doi:
                documents.append(abstract)
                metadatas.append({
                    "title": paper.get('title', 'N/A'),
                    "authors": ", ".join(paper.get('authors', [])[:3]),
                    "year": paper.get('year', 'N/A'),
                    "doi": doi
                })
                ids.append(doi)
                # --- NEW: Generate the embedding for the abstract ---
                embeddings.append(embed_text(abstract, use_cache=True).tolist())

        if not documents:
            print("WARNING: No documents with abstracts found to build RAG index.")
            return False

        # `upsert` can take embeddings directly. This is the key change.
        collection.upsert(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Successfully built RAG index with SciBERT. Indexed {len(documents)} papers.")
        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to build RAG index for collection '{collection_name}'. Reason: {e}")
        return False

def query_rag_index(collection_name: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Queries the RAG index with SciBERT to find relevant documents for a user's chat message.
    """
    print(f"INFO: Querying RAG index '{collection_name}' with query: '{query}'")
    try:
        # Get the collection without specifying an embedding function, as it's set on creation
        collection = client.get_collection(name=collection_name)

        # --- UPDATED QUERY LOGIC ---
        # 1. Manually create an embedding for the query using our consistent SciBERT model.
        query_embedding = embed_text(query, use_cache=True).tolist()

        # 2. Query the collection using the generated embedding, not the raw text.
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        # --- END OF UPDATE ---

        # The rest of the result processing remains the same
        retrieved_docs = []
        if results and results.get('documents') and results.get('metadatas'):
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            for doc, meta in zip(docs, metas):
                retrieved_docs.append({
                    'abstract': doc,
                    **meta # Unpack the metadata (title, authors, etc.)
                })

        print(f"✅ RAG query successful. Found {len(retrieved_docs)} relevant documents.")
        return retrieved_docs

    except Exception as e:
        print(f"❌ ERROR: Failed to query RAG index '{collection_name}'. Reason: {e}")
        # Return an empty list on failure so the app doesn't crash.
        return []