def rag_request(groq_api_key, user_query,
                     tfidf_top_n=5, chroma_top_n=5, final_top_k=3,
                     persist_dir=r"C:\profolders\Internships\Inceptai\rag\RA\chromadb_data",
                     collection_name="kql_context_embeddings"):
    """
    Runs the full retrieval and reranking pipeline and then passes the results
    to a ChatGroq LLM with a custom prompt. It returns the final response string.
    
    Parameters:
      groq_api_key (str): The GROQ API key.
      user_query (str): The user's input query.
      tfidf_top_n (int): Number of top TF-IDF results to retrieve.
      chroma_top_n (int): Number of top ChromaDB results to retrieve.
      final_top_k (int): Number of final top results after reranking.
      persist_dir (str): Persistence directory for ChromaDB.
      collection_name (str): Name of the ChromaDB collection.
      
    Returns:
      str: The final response from the LLM.
    """
    import os
    # Set the GROQ API key environment variable.
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Import retrieval and cleaning functions.
    from RA.rank_fusion import rank_fusion_rerank_string
    from RA.remove_think import remove_think_tags

    # Retrieve fused and reranked queries from the vector database.
    final_string = rank_fusion_rerank_string(
        user_query,
        tfidf_top_n=tfidf_top_n,
        chroma_top_n=chroma_top_n,
        final_top_k=final_top_k,
        persist_dir=persist_dir,
        collection_name=collection_name
    )

    # Import and initialize the ChatGroq LLM.
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # Add any additional parameters if needed.
    )

    # Construct the prompt template.
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that understands the KQL queries retrieved from a vector database. "
                "You are given top n queries; your goal is to determine which query is most relevant to the given input. "
                "The user query is: {user_query}. The following are the top {n} results from the vector database: {top_n_queries}"
            ),
            ("human", "{input}"),
        ]
    )

    # Combine the prompt with the LLM as a chain.
    chain = prompt | llm

    # Invoke the chain with the provided values.
    response = chain.invoke(
        {
            "user_query": user_query,
            "n": final_top_k,
            "top_n_queries": final_string,
            "input": user_query,
        }
    ).content

    # Clean the response.
    response = remove_think_tags(response)
    return response

#'''
# Example usage:
if __name__ == "__main__":
    groq_key = "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"
    user_query = "retrieve the total number of records in the StormEvents table"
    final_response = rag_request(groq_key, user_query,
                                      tfidf_top_n=5,
                                      chroma_top_n=5,
                                      final_top_k=3,
                                      persist_dir=r"C:\profolders\Internships\Inceptai\rag\RA\chromadb_data",
                                      collection_name="kql_context_embeddings")
    print("Final response from LLM:")
    print(final_response)
#'''
