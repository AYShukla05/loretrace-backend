from functools import lru_cache

from sentence_transformers import SentenceTransformer

# Self-hosted per LoreTrace_AI_Layer_Decision.md, chosen over text-embedding-3-small
# to avoid the one remaining OpenAI dependency in the stack. all-MiniLM-L6-v2 failed
# Gate 1 in LoreTrace_Quality_Gates.md (recall@5 58% vs. Voyage AI's 83% on the real
# corpus, LoreTrace_Gate1_Recall_Results.md) — replaced with bge-small-en-v1.5, a
# retrieval-tuned model at the same 384 dimensions, before reaching for a paid API.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# BGE models are asymmetric: this exact instruction prefix on the query side (and
# only the query side) is the model's own documented usage for retrieval, not an
# arbitrary choice - https://huggingface.co/BAAI/bge-small-en-v1.5.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    if is_query:
        texts = [QUERY_INSTRUCTION + text for text in texts]
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()
