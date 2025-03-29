def query_tf_idf(query, top_k):
    import pandas as pd
    import pickle
    from sklearn.metrics.pairwise import cosine_similarity

    # Load the vectorizer
    with open(r"C:\profolders\Internships\Inceptai\rag\RA\tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    print("TF-IDF vectorizer loaded.")

    # Load the original documents (if needed for display)
    df_docs = pd.read_csv(r"C:\profolders\Internships\Inceptai\rag\RA\original_documents.csv")
    print("Original documents loaded.")

    # Load the TF-IDF matrix CSV and convert back to a NumPy array
    tfidf_df = pd.read_csv(r"C:\profolders\Internships\Inceptai\rag\RA\tfidf_database.csv", index_col="RowIndex")
    tfidf_matrix = tfidf_df.values
    print("TF-IDF matrix loaded.")

    def retrieve_similar_documents(query, top_k=5):
        # Convert the query to a TF-IDF vector using the loaded vectorizer
        query_vector = tfidf_vectorizer.transform([query])
        
        # Compute cosine similarity between the query vector and all document vectors
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get the indices of the top-k similar documents
        top_indices = similarities.argsort()[::-1][:top_k]
        
        # Retrieve the corresponding rows from the original dataframe and similarity scores
        results = df_docs.iloc[top_indices].copy()
        results["similarity"] = similarities[top_indices]
        return results

    # Retrieve similar documents
    results = retrieve_similar_documents(query, top_k)
    print("Top similar entries found:")
    print(results)

    # Convert the results DataFrame to a list of dictionaries
    results_list = results.to_dict(orient="records")
    return results_list

#'''
# Example usage:
topk_results = query_tf_idf("1.	Check in last 30 days if the user account parth.test@pcsassure.me have been involved in any incident on Microsoft Sentinel.", 5)
print("Returned list of top-k results:")
print(topk_results)
#'''