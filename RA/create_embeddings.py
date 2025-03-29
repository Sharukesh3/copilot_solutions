import os
import shutil
import pandas as pd
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
import traceback
import chromadb

# Define the persistence directory path
persist_dir = "./chromadb_data_new"

# Delete the persistent directory if it exists to start fresh
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
    print(f"Cleared existing persistence directory: {persist_dir}")

# Ensure the directory is recreated
os.makedirs(persist_dir, exist_ok=True)
print(f"Created persistence directory: {persist_dir}")

# Set the environment variable for persistence (optional)
os.environ["CHROMA_DB_DIR"] = persist_dir

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path=persist_dir)
collection = chroma_client.get_or_create_collection(name="kql_context_embeddings")
print("ChromaDB initialized with persistence.")

# Check if ChromaDB is installed properly (this block is now redundant because we imported above)
use_chromadb = True
print("ChromaDB imported successfully")

# Load the dataframe from the augmented CSV
try:
    df = pd.read_csv(r"C:\profolders\Internships\Inceptai\rag\dataset\kql_provided_augmented.csv")
    print(f"Loaded {len(df)} rows from kql_augmented.csv")
    print(f"Columns in dataframe: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    traceback.print_exc()

# Initialize the embedding model (must be the same as used for querying)
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded successfully")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    traceback.print_exc()

# Define a function to process and store embeddings
def process_embeddings(df):
    if not use_chromadb:
        embeddings_data = []
    
    required_columns = ['context', 'kql']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"Error: Missing required columns: {missing}")
        return
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing embeddings"):
        try:
            context = str(row['context']) if pd.notna(row['context']) else ""
            kql = str(row['kql']) if pd.notna(row['kql']) else ""
            combined_text = f"{context} {kql}".strip()
            
            if not combined_text:
                print(f"Warning: Empty text for row {idx}, skipping")
                continue
            
            embedding = embedding_model.encode(combined_text).tolist()
            
            if use_chromadb:
                collection.add(
                    documents=[combined_text],
                    metadatas=[{"row_id": str(idx)}],
                    ids=[f"row_{idx}"],
                    embeddings=[embedding]
                )
                # Log every 10 rows
                if idx % 10 == 0:
                    print(f"Embedding for Row {idx} stored in ChromaDB.")
            else:
                embeddings_data.append({
                    'row_id': idx,
                    'combined_text': combined_text,
                    'embedding': embedding
                })
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            traceback.print_exc()
    
    if not use_chromadb:
        try:
            embeddings_df = pd.DataFrame(embeddings_data)
            embeddings_df.to_pickle("embeddings_data.pkl")
            print("Embeddings saved to 'embeddings_data.pkl'")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
            traceback.print_exc()

# Process embeddings: first test on a small subset then on the entire dataframe
print("Processing first 5 rows as a test...")
try:
    process_embeddings(df.head(5))
    print("Test successful!")
    
    # Optionally, clear the collection before processing all rows
    # Uncomment the following lines if you want to reset the collection:
    # chroma_client.delete_collection("kql_context_embeddings")
    # collection = chroma_client.get_or_create_collection(name="kql_context_embeddings")
    # print("ChromaDB collection reset for processing all rows.")
    
    print("Processing all rows...")
    process_embeddings(df)
except Exception as e:
    print(f"Error in processing: {e}")
    traceback.print_exc()

# Verify that the documents are saved
print(f"Total documents in collection: {collection.count()}")

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Assuming `df` is already defined and contains the 'context' and 'kql' columns
# Combine 'context' and 'kql' columns to create a corpus
df = pd.read_csv(r"C:\profolders\Internships\Inceptai\rag\dataset\kql_provided_augmented.csv")
corpus = df.apply(lambda row: f"{row['context']} {row['kql']}", axis=1)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the corpus to generate TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Convert the TF-IDF matrix to a DataFrame for better readability
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out(),
    index=df.index
)

# Save the TF-IDF DataFrame to a CSV file
tfidf_df.to_csv("tfidf_database.csv", index_label="RowIndex")
print("TF-IDF database saved to 'tfidf_database.csv'.")

