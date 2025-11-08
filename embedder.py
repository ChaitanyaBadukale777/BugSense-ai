# embedder.py
"""
Creates embeddings for texts and performs duplicate/similarity detection.
Uses sentence-transformers for embeddings and FAISS for fast search.
If faiss isn't available, falls back to sklearn NearestNeighbors.
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Tuple

# Try FAISS, fallback
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingIndex:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None  # numpy array
        self.texts = []
        self.index = None

    def encode(self, texts: List[str]):
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Ensure dtype and contiguity for downstream libraries (faiss expects float32)
        return np.ascontiguousarray(embs, dtype=np.float32)

    def build(self, texts: List[str]):
        self.texts = texts
        self.embeddings = self.encode(texts)
        if _HAS_FAISS:
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)  # inner product for cosine (after normalization)
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
        else:
            # sklearn fallback
            self.index = NearestNeighbors(n_neighbors=5, metric='cosine')
            self.index.fit(self.embeddings)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_emb = self.encode([text])[0]
        if _HAS_FAISS:
            # normalize the query vector in-place and search with the same array
            q_arr = np.ascontiguousarray(np.array([q_emb], dtype=np.float32))
            faiss.normalize_L2(q_arr)
            D, I = self.index.search(q_arr, top_k)
            # faiss returns distances as inner product; map to similarity
            sims = D[0].tolist()
            idxs = I[0].tolist()
            return [(i, float(s)) for i, s in zip(idxs, sims)]
        else:
            # sklearn returns distances; convert to similarity = 1 - distance
            distances, indices = self.index.kneighbors([q_emb], n_neighbors=top_k)
            dists = distances[0]
            idxs = indices[0]
            return [(int(i), float(1.0 - d)) for i, d in zip(idxs, dists)]

    def find_duplicates(self, threshold: float = 0.75) -> List[Tuple[int, int, float]]:
        """
        Return pairs (i, j, similarity) for similarity > threshold
        """
        pairs = []
        if self.embeddings is None:
            return pairs
        if _HAS_FAISS:
            # ensure embeddings are float32 contiguous and normalized for search
            emb_arr = np.ascontiguousarray(self.embeddings.astype(np.float32))
            faiss.normalize_L2(emb_arr)
            D, I = self.index.search(emb_arr, 6)  # self-neighbors included
            for i, (idxs, sims) in enumerate(zip(I, D)):
                for idx, sim in zip(idxs[1:], sims[1:]):  # skip itself at position 0
                    if sim >= threshold:
                        if i < idx:
                            pairs.append((i, int(idx), float(sim)))
        else:
            # compute cosine similarities matrix (small-scale)
            sims = util.cos_sim(self.embeddings, self.embeddings).numpy()
            n = sims.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if sims[i, j] >= threshold:
                        pairs.append((i, j, float(sims[i, j])))
        return pairs
