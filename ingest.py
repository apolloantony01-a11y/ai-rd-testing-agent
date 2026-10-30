#!/usr/bin/env python3
"""W2 ingestion: 采集 -> 切分 -> 去重 -> 版本化 -> 本地向量入库.

无 Docker 可跑: 只依赖 Python 标准库 + numpy(本机已有).
向量与 embedding 都是可替换的适配器, W4 要换 pgvector 时只换 Store,
不改切分/去重/版本化逻辑.

用法:
  python3 ingest.py --source <模板引擎目录> --source <week1文档> \
      --source <week3样本> --out ./store
  python3 ingest.py --source ... --out ./store --embedder tfidf   # 默认
  python3 ingest.py --source ... --out ./store --embedder st      # 需装 sentence-transformers
"""
import argparse, hashlib, json, math, os, re, sys
from pathlib import Path

import numpy as np

CHUNK_MAX = 600   # 文档类: 400-600 字一片(选型4默认值)
CHUNK_MIN = 400
OVERLAP = 100     # 保留一小段上下文, 防切断


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_markdown(text: str):
    """按段落累积, 超 600 字则切, 下一片带 100 字 overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 1 > CHUNK_MAX:
            chunks.append(cur)
            cur = cur[-OVERLAP:] + "\n" + p if len(cur) > OVERLAP else p
            if len(cur) > CHUNK_MAX:  # 单段本身超长, 硬切
                while len(cur) > CHUNK_MAX:
                    chunks.append(cur[:CHUNK_MAX])
                    cur = cur[CHUNK_MAX - OVERLAP:]
        else:
            cur = (cur + "\n" + p).strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def split_python(text: str):
    """按函数/类切分, 单函数超长再按行硬切(每片约 150 行)."""
    starts = [m.start() for m in re.finditer(r"(?m)^(def |class |async def )", text)]
    if not starts:
        return [text] if text.strip() else []
    starts.append(len(text))
    chunks = []
    for a, b in zip(starts, starts[1:]):
        piece = text[a:b].strip()
        if not piece:
            continue
        lines = piece.splitlines()
        if len(lines) <= 150:
            chunks.append(piece)
        else:
            for i in range(0, len(lines), 150):
                chunks.append("\n".join(lines[i:i + 150]))
    return chunks


def split_json(text: str):
    try:
        obj = json.loads(text)
    except Exception:
        return [text]
    if isinstance(obj, dict):
        return [json.dumps({k: v}, ensure_ascii=False) for k, v in obj.items()]
    return [text]


def chunk_file(path: Path, rel: str):
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    suffix = path.suffix.lower()
    if suffix == ".md":
        pieces = split_markdown(text)
        kind = "doc"
    elif suffix == ".py":
        pieces = split_python(text)
        kind = "code"
    elif suffix == ".json":
        pieces = split_json(text)
        kind = "config"
    else:
        return []
    out = []
    for i, piece in enumerate(pieces):
        title = piece.strip().splitlines()[0][:80] if piece.strip() else ""
        out.append({
            "source": rel, "kind": kind, "ordinal": i, "title": title,
            "text": piece, "content_hash": sha(piece),
        })
    return out


SKIP_DIRS = {".venv", "venv", ".git", "node_modules", "__pycache__",
             "store", ".tox", ".pytest_cache", ".mypy_cache"}


def collect(sources):
    """sources 可为目录或单个文件; 只收 .md/.py/.json 文本.

    跳过虚拟环境、版本库、依赖目录和隐藏目录: 2026-10-30 Mac 实测,
    template-engine 下的 .venv 曾贡献 589 个第三方库文件, 污染知识库,
    旧版只排 __pycache__ 是漏网之源, 故改为按目录名 + 隐藏目录双重排除.
    """
    files = []
    for s in sources:
        p = Path(s).expanduser()
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                if any(part in SKIP_DIRS or part.startswith(".")
                       for part in f.parts):
                    continue
                if f.suffix.lower() in (".md", ".py", ".json"):
                    files.append(f)
        else:
            print(f"警告: 来源不存在, 跳过: {s}", file=sys.stderr)
    return files


# ---------- Embedding: TF-IDF 基线(默认) / sentence-transformers(可选) ----------

def tokenize(text: str):
    toks = []
    for w in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text):
        if re.match(r"^[\u4e00-\u9fff]+$", w):
            toks.append(w)
            toks += [w[i:i + 2] for i in range(len(w) - 1)]  # 中文二元字
        else:
            toks.append(w.lower())
    return toks


class TfidfEmbedder:
    def __init__(self):
        self.vocab, self.idf = {}, None

    def fit(self, texts):
        from collections import Counter
        df = Counter()
        for t in texts:
            for tok in set(tokenize(t)):
                df[tok] += 1
        vocab = {tok: i for i, tok in enumerate(sorted(df))}
        n = len(texts)
        idf = np.array([math.log((n + 1) / (df[tok] + 1)) + 1.0
                        for tok in sorted(df)], dtype=np.float32)
        self.vocab, self.idf = vocab, idf
        return self

    def encode(self, texts):
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for r, t in enumerate(texts):
            for tok in tokenize(t):
                j = self.vocab.get(tok)
                if j is not None:
                    mat[r, j] += 1.0
            if self.idf is not None:
                mat[r] *= self.idf
            norm = np.linalg.norm(mat[r])
            if norm:
                mat[r] /= norm
        return mat


def load_embedder(name: str, texts):
    if name == "st":
        try:
            from sentence_transformers import SentenceTransformer
            model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
            print(f"embedding: sentence-transformers {model}")
            return ("st", SentenceTransformer(model))
        except Exception as e:
            print(f"警告: sentence-transformers 不可用({e}), 自动降级为 tfidf 基线",
                  file=sys.stderr)
    print("embedding: tfidf 基线(无模型下载, 仅作骨架评测; 语义召回待装 bge 后重跑对比)")
    emb = TfidfEmbedder().fit(texts)
    return ("tfidf", emb)


def encode_all(kind, obj, texts):
    if kind == "st":
        vecs = obj.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    return obj.encode(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embedder", default="tfidf", choices=["tfidf", "st"])
    args = ap.parse_args()

    files = collect(args.source)
    # 来源相对名: 以各自 source 根为基准, 单文件则用文件名
    new_chunks = []
    for f in files:
        rel = f.name
        # 尽量保留 M1-x/ 前缀以便评测定位
        parts = f.parts
        if "M1-" in str(f):
            idx = next(i for i, p in enumerate(parts) if p.startswith("M1-"))
            rel = "/".join(parts[idx:])
        new_chunks.extend(chunk_file(f, rel))
    print(f"采集文件: {len(files)} 个, 切分 chunk: {len(new_chunks)} 个")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "chunks.jsonl"
    old = []
    if manifest_path.exists():
        old = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    old_by_key = {(c["source"], c["ordinal"]): c for c in old if c.get("status") == "current"}

    kept, deduped = [], 0
    seen_hash = set()
    for c in new_chunks:
        key = (c["source"], c["ordinal"])
        if c["content_hash"] in seen_hash:
            deduped += 1
            continue
        seen_hash.add(c["content_hash"])
        prev = old_by_key.get(key)
        if prev and prev["content_hash"] == c["content_hash"]:
            kept.append(prev)  # 内容未变: 复用旧记录, 不重复入库
            continue
        c.update({"chunk_id": f"{c['source']}#{c['ordinal']}#{c['content_hash'][:12]}",
                  "status": "current"})
        kept.append(c)
    # 版本化: 旧 current 但新一轮已不在的 source/ordinal, 标 superseded 保留
    new_keys = {(c["source"], c["ordinal"]) for c in kept}
    superseded = [dict(c, status="superseded") for c in old
                  if c.get("status") == "current" and (c["source"], c["ordinal"]) not in new_keys]
    print(f"去重跳过: {deduped} 个, 复用未变: "
          f"{sum(1 for c in kept if 'chunk_id' in c and c in old)} 个, "
          f" superseded: {len(superseded)} 个")

    current = [c for c in kept if c.get("status", "current") == "current"]
    kind, emb = load_embedder(args.embedder, [c["text"] for c in current])
    vecs = encode_all(kind, emb, [c["text"] for c in current])
    np.savez(out / "vectors.npz", vectors=vecs,
             vocab=json.dumps(getattr(emb, "vocab", {}), ensure_ascii=False)
             if kind == "tfidf" else "")
    if kind == "tfidf":
        np.save(out / "idf.npy", emb.idf)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for c in current + superseded:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump({"embedder": kind, "model": os.environ.get("EMBEDDING_MODEL", "")
                   if kind == "st" else "tfidf-char-bigram",
                   "chunks_current": len(current), "files": len(files)},
                  fh, ensure_ascii=False, indent=2)
    print(f"入库完成: {len(current)} 个 current chunk -> {out}")


if __name__ == "__main__":
    main()
