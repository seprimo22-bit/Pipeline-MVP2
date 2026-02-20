import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGEngine:

    def __init__(self, index_path="vector.index"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_path = index_path
        self.index = None
        self.documents = []

        if os.path.exists(index_path):
            self.load_index()

    # -----------------------------
    # BUILD INDEX
    # -----------------------------
    def build_index(self, documents):
        """
        documents: list of text chunks
        """
        self.documents = documents
        embeddings = self.model.encode(documents)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

        faiss.write_index(self.index, self.index_path)

    # -----------------------------
    # LOAD EXISTING INDEX
    # -----------------------------
    def load_index(self):
        self.index = faiss.read_index(self.index_path)

    # -----------------------------
    # RETRIEVE TOP MATCHES
    # -----------------------------
    def retrieve(self, query, top_k=3):
        if self.index is None:
            return []

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])

        return results
