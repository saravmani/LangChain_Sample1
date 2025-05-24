import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

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
    docs = load_documents(md_files)  
    print(docs[0].metadata)
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs) 
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )
    ids = vector_store.add_documents(documents=all_splits)
    results = vector_store.similarity_search_with_score("User Update API")
    print(results[0]) 
 



from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)