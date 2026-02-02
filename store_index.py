from dotenv import load_dotenv
import os
import time
from tqdm import tqdm
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document  # <-- CHANGED
import uuid

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found!")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Step 1: Load PDF
print("Loading PDF...")
extracted_data = load_pdf_files(data='data/')
print(f"✓ Loaded {len(extracted_data)} pages")

# Step 2: Filter
filter_data = filter_to_minimal_docs(extracted_data)
print(f"✓ Filtered {len(filter_data)} documents")

# Step 3: Split
text_chunks = text_split(filter_data)
print(f"✓ Created {len(text_chunks)} chunks")

if not text_chunks:
    raise ValueError("No chunks created! Check your PDF.")

# Step 4: Embeddings
print("Loading embeddings model...")
embeddings = download_hugging_face_embeddings()

# Step 5: Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'medical-chatbot'

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud="aws", region='us-east-1')
    )
    # Wait for index ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    print("✓ Index ready")
else:
    print(f"Index exists: {index_name}")

index = pc.Index(index_name)

# Step 6: MANUAL UPLOAD (Reliable method)
print(f"\nUploading {len(text_chunks)} chunks to Pinecone...")

batch_size = 50  # Small batches for reliability
total_uploaded = 0

for i in tqdm(range(0, len(text_chunks), batch_size)):
    batch = text_chunks[i:i+batch_size]
    
    # Create embeddings for batch
    texts = [doc.page_content for doc in batch]
    metadatas = [doc.metadata for doc in batch]
    
    # Generate embeddings
    vectors = embeddings.embed_documents(texts)
    
    # Prepare upsert data
    upsert_data = []
    for j, (vec, metadata) in enumerate(zip(vectors, metadatas)):
        # Unique ID for each chunk
        doc_id = f"doc_{i+j}_{uuid.uuid4().hex[:8]}"
        upsert_data.append({
            'id': doc_id,
            'values': vec,
            'metadata': {
                'text': texts[j][:1000],  # Store first 1000 chars
                'source': metadata.get('source', 'unknown')
            }
        })
    
    # Upload to Pinecone
    try:
        index.upsert(vectors=upsert_data, namespace="")
        total_uploaded += len(batch)
    except Exception as e:
        print(f"\nError in batch {i//batch_size + 1}: {e}")
        time.sleep(2)  # Wait and retry once
        try:
            index.upsert(vectors=upsert_data, namespace="")
            total_uploaded += len(batch)
        except:
            print(f"Failed batch {i//batch_size + 1}, skipping...")

# Step 7: Verify
time.sleep(3)
stats = index.describe_index_stats()
print(f"\n{'='*50}")
print(f"UPLOAD COMPLETE!")
print(f"Total chunks processed: {len(text_chunks)}")
print(f"Successfully uploaded: {total_uploaded}")
print(f"Vectors in index: {stats.total_vector_count}")
print(f"Namespaces: {list(stats.namespaces.keys())}")
print(f"{'='*50}")