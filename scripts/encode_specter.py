from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

from transformers import AutoModel, AutoTokenizer

from scirelay.benchmark import load_dataset


DEFAULT_MODEL = "sentence-transformers/allenai-specter"
DEFAULT_REVISION = "2c68eeca61259b2dd70c3f2628219f925df7031a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode SciFact with frozen SPECTER")
    parser.add_argument("--data", type=Path, default=Path("data/raw/scifact-beir"))
    parser.add_argument("--output", type=Path, default=Path("results/dense/cache/specter"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def encode(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            cls = model(**encoded).last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        rows.append(cls.detach().cpu().numpy().astype(np.float32, copy=False))
        print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(rows, axis=0)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dataset = load_dataset(args.data)
    query_ids = sorted(dataset.queries, key=int)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(args.model, revision=args.revision)
    device = torch.device(args.device)
    model.eval().to(device)
    sep = tokenizer.sep_token or "[SEP]"
    document_texts = [
        f"{document.title} {sep} {document.text}".strip()
        for document in dataset.documents
    ]
    query_texts = [dataset.queries[query_id].text for query_id in query_ids]

    document_embeddings = encode(
        document_texts, tokenizer, model, args.batch_size, args.max_length, device
    )
    query_embeddings = encode(
        query_texts, tokenizer, model, args.batch_size, args.max_length, device
    )
    np.save(args.output / "document_embeddings.npy", document_embeddings, allow_pickle=False)
    np.save(args.output / "query_embeddings.npy", query_embeddings, allow_pickle=False)
    np.save(
        args.output / "document_ids.npy",
        np.asarray([document.doc_id for document in dataset.documents]),
        allow_pickle=False,
    )
    np.save(args.output / "query_ids.npy", np.asarray(query_ids), allow_pickle=False)
    metadata = {
        "model_id": args.model,
        "revision": args.revision,
        "pooling": "L2-normalized final-layer [CLS] embedding",
        "document_format": "title [SEP] abstract",
        "query_format": "claim text",
        "max_length": args.max_length,
        "hidden_size": int(document_embeddings.shape[1]),
        "documents": len(document_embeddings),
        "queries": len(query_embeddings),
        "dtype": str(document_embeddings.dtype),
        "device": str(device),
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
