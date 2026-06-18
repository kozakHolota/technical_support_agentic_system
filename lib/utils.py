import json
from functools import cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_jsons(knowledge_base_dir: str = "knowledge_base") -> list[Document]:
    """Get knowledge base documents from JSON files."""
    documents: list[Document] = []
    for json_path in Path(knowledge_base_dir).glob("**/*.json"):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                documents.append(Document(page_content=json.dumps(item, ensure_ascii=False)))
        else:
            documents.append(Document(page_content=json.dumps(data, ensure_ascii=False)))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

@cache
def get_faiss(knowledge_base_dir: str = "knowledge_base") -> FAISS:
    """Get faiss documents from JSON files."""
    chunks = load_jsons(knowledge_base_dir)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

def search_query(query: str, knowledge_base_dir: str = "knowledge_base") -> list[Document]:
    """Search faiss documents from JSON files."""
    faiss = get_faiss(knowledge_base_dir)
    return faiss.similarity_search(query, k=4)
