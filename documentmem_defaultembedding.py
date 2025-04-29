import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils import embedding_functions



load_dotenv()

def find_md_files(root_dir):
    md_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.md'):
                md_files.append(os.path.join(dirpath, filename))
    return md_files

def load_documents(md_files):
    docs = []
    for file_path in md_files:
        loader = TextLoader(file_path)
        docs.extend(loader.load())
    return docs





if __name__ == "__main__":
    root_dir = "docs"
    md_files = find_md_files(root_dir)
    print(f"Found {len(md_files)} markdown files.")
    documents = load_documents(md_files)  
    

    client = chromadb.PersistentClient(path="./chroma_pure_db")

    # 2. Create or get collection
    collection_name = "example_collection"
    try:
        collection = client.get_collection(name=collection_name)
    except chromadb.errors.InvalidCollectionException:
        collection = client.create_collection(name=collection_name)

    # 3. Prepare documents to insert
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [f"doc_{i}" for i in range(len(documents))]

    # 4. Add documents (Chroma will auto-embed)
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    # 5. Search example
    query = "User Update API"
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
 
    for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
        print(f"\nResult {i+1}:")
        print(f"Distance: {dist}")
        print(f"Metadata: {meta}")
        print(f"Document snippet: {doc[:200]}...")

    print("\nTop 3 matching documents:")