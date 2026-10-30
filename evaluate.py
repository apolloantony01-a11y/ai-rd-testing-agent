#!/usr/bin/env python3
"""20 条查询召回评测: 命中 = top-k 内任一 chunk 来源含 expect_source,
或文本含任一 keywords. 命中标准写在报告里, 不只报通过率.

用法: python3 evaluate.py --store ./store --queries queries.json --topk 5
"""
import argparse, json
from pathlib import Path

from search import search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    queries = json.loads(Path(args.queries).read_text())
    hits, failures = 0, []
    for item in queries:
        res = search(args.store, item["q"], topk=args.topk, rerank=True)
        hit = any(item["expect_source"] in r["chunk"]["source"]
                  or any(k in r["chunk"]["text"] for k in item["keywords"])
                  for r in res)
        if hit:
            hits += 1
        else:
            top = "; ".join(f"{r['chunk']['source']}#{r['chunk']['ordinal']}"
                            for r in res[:3])
            failures.append({"q": item["q"], "expect": item["expect_source"],
                             "top3": top})
    total = len(queries)
    print(f"命中率 top-{args.topk}: {hits}/{total} = {hits/total:.0%}")
    print(f"失败 {len(failures)} 条:")
    for f in failures:
        print(f"  - {f['q']} (期望 {f['expect']}) top3: {f['top3']}")
    report = {"topk": args.topk, "hits": hits, "total": total,
              "rate": hits / total, "failures": failures,
              "note": "当前为 tfidf 基线 + 规则重排版, 非语义 embedding 版成绩"}
    Path("evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告已写 evaluation_report.json (运行目录)")


if __name__ == "__main__":
    main()
