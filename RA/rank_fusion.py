def rank_fusion_rerank_string(query, tfidf_top_n=5, chroma_top_n=5, final_top_k=3,
                              persist_dir=r"C:\profolders\Internships\Inceptai\rag\RA\chromadb_data", collection_name="kql_context_embeddings"):
    """
    Combines TF-IDF and ChromaDB retrieval with rank fusion and reranks the fused candidates,
    then returns a formatted string with the top results.
    
    The process is:
      1. Retrieve candidates from TF-IDF and ChromaDB.
      2. Convert TF-IDF results (dictionaries) into strings (by concatenating 'context' and 'kql').
      3. Fuse the candidate lists (removing duplicates).
      4. Use a cross-encoder to rerank the candidates.
      5. Format and return the top final_top_k results as a single string.
    """
    # Import necessary modules
    from sentence_transformers import CrossEncoder, SentenceTransformer
    import numpy as np
    # Import your retrieval functions. Make sure these functions are in your PYTHONPATH.
    from RA.query_tf_idf import query_tf_idf
    from RA.query_chroma_db import query_chroma_db

    # Step 1: Retrieve candidates.
    tfidf_results = query_tf_idf(query, top_k=tfidf_top_n)
    print(persist_dir)
    chroma_candidates = query_chroma_db(persist_dir, collection_name, query, top_k=chroma_top_n)
    
    # Step 2: Convert TF-IDF dictionaries to strings.
    tfidf_candidates = []
    for res in tfidf_results:
        context = res.get("context", "").strip()
        kql = res.get("kql", "").strip()
        combined = f"{context} {kql}".strip()
        if combined:
            tfidf_candidates.append(combined)
    
    # Step 3: Fuse candidate lists (remove duplicates).
    fused_candidates = list(set(tfidf_candidates + chroma_candidates))
    
    # Step 4: Rerank candidates using a cross-encoder.
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, candidate) for candidate in fused_candidates]
    scores = reranker.predict(pairs)
    
    # Combine candidates with scores and sort descending.
    ranked_candidates = sorted(zip(fused_candidates, scores), key=lambda x: x[1], reverse=True)
    top_results = ranked_candidates[:final_top_k]
    
    # Step 5: Format the top results as a single string.
    output_lines = []
    for idx, (doc, score) in enumerate(top_results, 1):
        output_lines.append(f"Rank {idx}: Score = {score:.4f} | Query: {doc}")
        output_lines.append("")  # Add an extra blank line for spacing.
    result_string = "\n".join(output_lines).strip()
    
    return result_string

'''
# Example usage:
if __name__ == "__main__":
    user_query = "retrieve the total number of records in the StormEvents table"
    final_string = rank_fusion_rerank_string(user_query,
                                             tfidf_top_n=5,
                                             chroma_top_n=5,
                                             final_top_k=3,
                                             persist_dir="./chromadb_data",
                                             collection_name="kql_context_embeddings")
    print("Final top results after rank fusion and reranking:")
    print(final_string)
'''