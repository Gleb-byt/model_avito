import json
import random
from pathlib import Path
from typing import Dict, List, Any

DEFAULT_SYSTEM_PROMT = (
    "Ты — виртуальный ассистент по адаптации новых сотрудников Авито. "
    "Отвечай вежливо, структурированно и строго следуй предоставленной информации."
)


def load_jsonl(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """Loads a Json file into a dictionary keyed by the id field"""

    records = dict()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                records[item["id"]] = item
    return records


def prepare_dataset(
    queries_path: str = "data/raw/queries.jsonl",
    answers_path: str = "data/raw/answers.jsonl",
    output_dir: str = "data/processed",
    val_ratio: float = 0.1,
    group_by_intent: bool = False,
    seed: int = 42,
    system_prompt: str = DEFAULT_SYSTEM_PROMT,
):
    """
    This function merges two files and creates one .jsonl train/val splits
    Which allow to work with the data as one objecet
    """

    random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    queries: Dict[str, Dict[str, Any]] = load_jsonl(Path(queries_path))
    answers: Dict[str, Dict[str, Any]] = load_jsonl(Path(answers_path))

    """To match data correctly i should find common ids"""

    common_ids: List[str] = sorted(set(queries.keys()) & set(answers.keys()))

    print(f"Len common ids: {common_ids.__len__()} out of {len(queries)}")

    merged_data: List[Dict[str, Any]] = list()

    for item_id in common_ids:
        q_item = queries[item_id]
        a_item = answers[item_id]

        merged_data.append(
            {
                "id": item_id,
                "intent_id": q_item.get("intent_id", a_item.get("intent_id")),
                "category": q_item.get("category"),
                "grounding": a_item.get("grounding"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q_item["query"]},
                    {"role": "assistant", "content": a_item["answer"]},
                ],
            }
        )

    if group_by_intent:
        unique_intents = list({item["intent_id"] for item in merged_data})
        random.shuffle(unique_intents)
        val_count = int(len(unique_intents) * val_ratio)
        val_intents = set(unique_intents[:val_count])

        train_set = [d for d in merged_data if d["intent_id"] not in val_intents]
        val_set = [d for d in merged_data if d["intent_id"] in val_intents]

    else:
        random.shuffle(merged_data)
        split_idx = int(len(merged_data) * (1.0 - val_ratio))
        train_set = merged_data[:split_idx]
        val_set = merged_data[split_idx:]

    train_file = out_path / "train_pairs.jsonl"
    val_file = out_path / "val_pairs.jsonl"

    for filepath, dataset in [(train_file, train_set), (val_file, val_set)]:
        with open(filepath, "w", encoding="utf-8") as f:
            for i in dataset:
                f.write(json.dumps(i, ensure_ascii=False) + "\n")

    print("file processed and saved in processed files directory")


if __name__ == "__main__":
    prepare_dataset(
        queries_path="data/raw/queries.jsonl",
        answers_path="data/raw/answers.jsonl",
        output_dir="data/processed",
        val_ratio=0.1,
        group_by_intent=False,
    )
