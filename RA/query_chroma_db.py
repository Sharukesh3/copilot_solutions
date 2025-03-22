def query_chroma_db(persist_dir, collection_name, query, top_k):
    import os
    import chromadb
    from sentence_transformers import SentenceTransformer

    # Set the environment variable for persistence
    os.environ["CHROMA_DB_DIR"] = persist_dir

    # Initialize the persistent client
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    # Load or create your collection
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Verify by printing the number of documents stored
    print(f"Collection loaded. Total documents: {collection.count()}")

    # Load the embedding model (ensure it's the same model used during data insertion)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def search_chroma_db(query, top_k=5):
        # Convert the user query into an embedding
        query_embedding = embedding_model.encode([query])
        
        # Perform the search in ChromaDB
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Extract the original texts from the search results
        # Note: results['documents'] is assumed to be a list of lists
        similar_texts = results['documents'][0]
        return similar_texts

    # Retrieve the top-k similar documents
    top_k_results = search_chroma_db(query, top_k=top_k)
    print(f"Top-{len(top_k_results)} similar entries found in ChromaDB:")
    for idx, result in enumerate(top_k_results, 1):
        print(f"Result {idx}: {result}")
        
    # Return the list of results
    return top_k_results

# Example usage:
'''
results = query_chroma_db(
    "./chromadb_data",
    "kql_context_embeddings",
    "retrieve the total number of records in the StormEvents table",
    5
)
print("Returned list of top-k results:")
print(results)
'''