#!/usr/bin/env python3
"""未见改写查询评测(与 queries.json 的 20 条已知查询区分开).

严格点: queries_unseen.json 里 keywords 为空, 命中只看 top-k 内 chunk
来源是否含 expect_source, 不用关键词兜底, 所以成绩不能直接和
20 条已知查询(带关键词兜底)的 20/20、19/20 对等比较, 只能看差距方向.

用法: python3 evaluate_unseen.py --store ./store --queries queries_unseen.json --topk 5
报告写 evaluation_report_unseen.json, 不动 evaluation_report.json.
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
    hits1, hits, failures = 0, 0, []
    for item in queries:
        res = search(args.store, item["q"], topk=args.topk, rerank=True)
        hit1 = bool(res) and item["expect_source"] in res[0]["chunk"]["source"]
        hit = any(item["expect_source"] in r["chunk"]["source"] for r in res)
        hits1 += hit1
        hits += hit
        if not hit:
            top = "; ".join(f"{r['chunk']['source']}#{r['chunk']['ordinal']}"
                            for r in res[:3])
            failures.append({"q": item["q"], "expect": item["expect_source"],
                             "top3": top})
    total = len(queries)
    print(f"未见查询 top-1: {hits1}/{total}, top-{args.topk}: {hits}/{total}")
    print(f"完全 miss {len(failures)} 条:")
    for f in failures:
        print(f"  - {f['q']} (期望 {f['expect']}) top3: {f['top3']}")
    report = {"top1": hits1, "topk": args.topk, "hits": hits, "total": total,
              "failures": failures,
              "note": "tfidf 基线 + 规则重排版, 严格来源命中(无关键词兜底), 非语义 embedding 版成绩"}
    Path("evaluation_report_unseen.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告已写 evaluation_report_unseen.json (运行目录)")


if __name__ == "__main__":
    main()
