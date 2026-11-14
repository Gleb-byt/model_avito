"""
The main idea in building index system
is that it is too hard to teach model certain data,
exspecially when we are speaking about fine tuning using
LoRA. For that i should use systems like RAG which allows
to find information based on the question. And information
which was found should be passed into model as input
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorKnowledgeBase:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        index_dir: str = "data/index",
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = SentenceTransformer(self.model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.documents: List[Dict[str, Any]] = list()

    def build_from_pairs(
        self,
        train_pairs_path: str = "data/processed/train_pairs.jsonl",
        deduplicate_by_intent: bool = False,
    ):
        """
        Extracts answer knowledge passages, encodes them, and adds them to FAISS
        """

        seen_intents = set()
        unique_docs: List[Dict[str, Any]] = list()

        with open(train_pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                intent_id = item.get("intent_id")

                assistant_text = ""
                for msg in item.get("messages", []):
                    if msg["role"] == "assistant":
                        assistant_text = msg["content"]
                        break

                if not assistant_text:
                    continue

                if deduplicate_by_intent and intent_id:
                    if intent_id in seen_intents:
                        continue
                    seen_intents.add(intent_id)

                unique_docs.append(
                    {
                        "id": item["id"],
                        "intent_id": intent_id,
                        "category": item.get("category"),
                        "grounding": item.get("grounding"),
                        "content": assistant_text,
                    }
                )
        self.documents = unique_docs
        print(f"Extracted {len(self.documents)} unique knowledge passages.")

        prefix = "passage: " if "e5" in self.model_name.lower() else ""
        passages = [prefix + doc["content"] for doc in self.documents]

        embeddings = self.embedder.encode(
            passages,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

        self.save()

    def save(self):
        index_file = self.index_dir / "faiss.index"
        docs_file = self.index_dir / "id2doc.pkl"
        faiss.write_index(self.index, str(index_file))
        with open(docs_file, "wb") as f:
            pickle.dump(self.documents, f)
        print(f"Saved FAISS index to {index_file} and docs map to {docs_file}")

    def load(self):
        index_file = self.index_dir / "faiss.index"
        docs_file = self.index_dir / "id2doc.pkl"
        self.index = faiss.read_index(str(index_file))
        with open(docs_file, "rb") as f:
            self.documents = pickle.load(f)
        print(f"Loaded {self.index.ntotal} vectors from {index_file}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieves top_k relevant documents for a given user query.
        """
        if self.index is None:
            raise ValueError(
                "Index not loaded. Call load() or build_from_pairs() first."
            )
        prefix = "query: " if "e5" in self.model_name.lower() else ""
        query_vec = self.embedder.encode(
            [prefix + query],
            normalize_embeddings=True,
        ).astype(np.float32)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results


if __name__ == "__main__":
    kb = VectorKnowledgeBase(model_name="intfloat/multilingual-e5-small")
    kb.build_from_pairs("data/processed/train_pairs.jsonl")
    # Quick test query
    test_query = "С чего мне начать стажировку в компании?"
    hits = kb.search(test_query, top_k=2)
    print(f"\nTest Query: {test_query}\nTop results:")
    for doc, score in hits:
        print(f"[{score:.4f}] {doc['content'][:120]}...")
