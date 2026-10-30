#!/usr/bin/env python3
"""召回 + 重排(规则版).

重排默认值说明: 选型4定为本地交叉编码器, 但本机未装模型时自动降级
为规则重排(词重叠 + 标题命中), 评测报告必须标注当前是规则重排版,
装好 bge/cross-encoder 后重跑同一评测做对比, 不把规则版成绩当成语义版.

用法: python3 search.py --store ./store --query "文字含单引号渲染失败" --topk 5
"""
import argparse, json, re
from pathlib import Path

import numpy as np

from ingest import tokenize, TfidfEmbedder


def load_store(store: Path):
    chunks = [json.loads(l) for l in (store / "chunks.jsonl").read_text().splitlines()
              if l.strip()]
    current = [c for c in chunks if c.get("status") == "current"]
    data = np.load(store / "vectors.npz", allow_pickle=True)
    vecs = data["vectors"]
    meta = json.loads((store / "meta.json").read_text())
    return current, vecs, meta


def search(store: str, query: str, topk: int = 10, rerank: bool = True):
    store = Path(store)
    chunks, vecs, meta = load_store(store)
    if meta["embedder"] == "tfidf":
        vocab = json.loads(str(np.load(store / "vectors.npz", allow_pickle=True)["vocab"]))
        emb = TfidfEmbedder()
        emb.vocab = vocab
        emb.idf = np.load(store / "idf.npy")
        qvec = emb.encode([query])[0]
    else:
        from sentence_transformers import SentenceTransformer
        import os
        model = SentenceTransformer(os.environ.get("EMBEDDING_MODEL",
                                                    "BAAI/bge-small-zh-v1.5"))
        qvec = np.asarray(model.encode([query], normalize_embeddings=True)[0],
                          dtype=np.float32)
    scores = vecs @ qvec
    order = np.argsort(-scores)[:topk].tolist()
    results = [{"score": float(scores[i]), "chunk": chunks[i]} for i in order]
    if rerank:
        qtoks = set(tokenize(query))
        for r in results:
            ctoks = set(tokenize(r["chunk"]["text"]))
            overlap = len(qtoks & ctoks) / max(len(qtoks), 1)
            title_hit = 1.0 if any(t in r["chunk"]["title"] for t in qtoks if len(t) > 1) else 0.0
            r["rerank_score"] = 0.7 * r["score"] + 0.3 * overlap + 0.05 * title_hit
        results.sort(key=lambda r: -r["rerank_score"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()
    for i, r in enumerate(search(args.store, args.query, args.topk,
                                 rerank=not args.no_rerank), 1):
        c = r["chunk"]
        print(f"{i}. [{c['source']}#{c['ordinal']}] score={r['score']:.3f} "
              f"rerank={r.get('rerank_score', 0):.3f} {c['title']}")
        print("   " + c["text"][:200].replace("\n", " "))


if __name__ == "__main__":
    main()
