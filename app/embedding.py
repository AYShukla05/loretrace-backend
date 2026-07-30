from functools import lru_cache

from sentence_transformers import SentenceTransformer

# Self-hosted per LoreTrace_AI_Layer_Decision.md, chosen over text-embedding-3-small
# to avoid the one remaining OpenAI dependency in the stack. Not final until it
# clears Gate 1 in LoreTrace_Quality_Gates.md (recall@k against a real corpus).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()
