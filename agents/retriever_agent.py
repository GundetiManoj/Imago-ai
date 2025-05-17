# agents/retriever_agent.py
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, util
import numpy as np
class RetrieverAgent:
    def __init__(self, corpus: List[Dict], model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the retriever agent with a corpus of documents/chunks.

        :param corpus: List of dicts, each with keys 'id' and 'text'
        :param model_name: SentenceTransformer model name for embeddings
        """
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = self._embed_corpus()

    def _embed_corpus(self) -> np.ndarray:
        texts = [doc['text'] for doc in self.corpus]
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the top_k most relevant chunks for the query.

        :param query: User query string
        :param top_k: Number of top chunks to retrieve
        :return: List of retrieved document chunks (dicts)
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)[0]

        retrieved_chunks = [self.corpus[hit['corpus_id']] for hit in hits]
        return retrieved_chunks