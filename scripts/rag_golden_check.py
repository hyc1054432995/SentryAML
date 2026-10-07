"""
SentryAML - RAG 层 R4：评测问题集的校验与解析
==============================================
评测集 config/rag_golden.yaml 是人写的，这个脚本负责"人写的东西对不对"：

  1. 格式：编号唯一、类型/语言/切分取值合法、必填字段齐全
  2. 标准答案：
       - gold / also_ok / near_miss 的条款，必须在该题调查日期（as_of）有效的版本里存在
       - answer_keys（答案要点）必须逐字出现在 gold 条款原文里 —— 保证"标准答案确实来自原文"
       - must_not（版本陷阱）不能出现在当期 gold 原文里，并且应当出现在同一条款的其他版本里，
         否则这个陷阱是假的
       - 版本题的 gold 条款，在其他版本里必须内容不同或不存在，否则不是真正的版本题
  3. 无答案题：gold 为空，必须写明 why（为什么知识库答不了）
  4. 切分纪律：同一条款不能同时出现在开发集和测试集的 gold 里；同一 group 的题必须在同一侧。
     否则在开发集上调参时，等于提前看到了测试集的答案所在位置
  5. 统计：题型 × 切分、语言、问题与原文的词面重合度（重合度过高的题偏向关键词检索，需要知道比例）

最后把每道题的 gold 解析到具体文件版本和检索块，写到 reports/rag_golden_resolved.json，
R6 的检索评测直接读这个文件。

用法：
    python scripts/rag_golden_check.py                                  # v1
    python scripts/rag_golden_check.py --golden config/rag_golden_v2.yaml --out reports/rag_golden_v2_resolved.json
v2 起每题多一个 pattern 字段（检索一次答不好的情形），并会列出与 v1 gold 重合的条款。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import yaml

from db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = PROJECT_ROOT / "config" / "rag_golden.yaml"
OUT = PROJECT_ROOT / "reports" / "rag_golden_resolved.json"

PATTERNS = ["bridge", "adjacent", "paraphrase", "two_part", "cross_doc", "version", "premise", "unanswerable"]
TYPES = ["lookup", "definition", "exact_id", "typology", "str_filing", "multi_hop", "version", "unanswerable"]
STOP = set("""a an the of to in on for and or is are be was were by with as at from that this which what when
where who whom how why does do did can could should would must may might it its their there than then into
under about any all some such if not no yes i my me we our you your bank banks s""".split())


def norm(s: str) -> str:
    s = re.sub(r"\[\^\d+\]", "", s)
    s = s.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "―": "-", "–": "-"}))
    return re.sub(r"\s+", " ", s).strip().lower()


def tokens(s: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z][a-z\-]+", norm(s)):
        w = w.strip("-")
        if len(w) < 3 or w in STOP:
            continue
        out.add(re.sub(r"(ies|es|s|ing|ed)$", "", w) or w)       # 极简词干，只用于统计重合度
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="评测问题集的校验与解析")
    ap.add_argument("--golden", default=str(GOLDEN))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    golden, out_path = Path(a.golden), Path(a.out)
    with open(golden, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    default_as_of = cfg["default_as_of"]
    qs = cfg["questions"]

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, family, effective_from, effective_to FROM kb_documents")
        docs = cur.fetchall()
        cur.execute("SELECT doc_id, unit_id, text FROM kb_units")
        units = {(d, u): t for d, u, t in cur.fetchall()}
        cur.execute("SELECT chunk_id, doc_id, unit_ids, text FROM kb_chunks")
        chunks_of = defaultdict(list)
        for cid, d, uids, text in cur.fetchall():
            for u in uids:
                chunks_of[(d, u)].append((cid, text))
    versions = defaultdict(list)
    for d, fam, f, t in docs:
        versions[fam].append((d, f, t))

    def in_force(fam: str, as_of: date) -> str | None:
        for d, f, t in versions.get(fam, []):
            if f <= as_of and (t is None or as_of < t):
                return d
        return None

    errors, warnings = [], []
    resolved = []
    gold_split = defaultdict(set)
    group_split = defaultdict(set)

    ids = Counter(q["id"] for q in qs)
    errors += [f"编号重复：{k}" for k, n in ids.items() if n > 1]

    for q in qs:
        qid = q["id"]
        E = lambda m: errors.append(f"{qid}：{m}")
        for k in ("type", "lang", "question", "split"):
            if not q.get(k):
                E(f"缺少字段 {k}")
        if q.get("type") not in TYPES:
            E(f"未知题型 {q.get('type')}")
        if q.get("lang") not in ("en", "zh"):
            E(f"未知语言 {q.get('lang')}")
        if q.get("split") not in ("dev", "test"):
            E(f"切分只能是 dev / test")
        if "pattern" in q and q["pattern"] not in PATTERNS:
            E(f"未知 pattern {q['pattern']}")
        as_of = q.get("as_of", default_as_of)
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)

        def resolve(refs, label):
            out = []
            for r in refs or []:
                d = in_force(r["family"], as_of)
                if d is None:
                    E(f"{label} {r['family']} 在 {as_of} 没有有效版本")
                elif (d, r["unit_id"]) not in units:
                    E(f"{label} {r['family']}:{r['unit_id']} 在 {d}（{as_of} 有效）中不存在")
                else:
                    out.append({"family": r["family"], "unit_id": r["unit_id"], "doc_id": d})
            return out

        gold = resolve(q.get("gold"), "gold")
        also_ok = resolve(q.get("also_ok"), "also_ok")
        near_miss = resolve(q.get("near_miss"), "near_miss")
        keys, must_not = q.get("answer_keys") or [], q.get("must_not") or []
        answerable = bool(q.get("gold"))

        if not answerable:
            if not q.get("why"):
                E("无答案题必须写 why")
            if keys:
                E("无答案题不应有 answer_keys")
        elif len(gold) == len(q["gold"]):                 # gold 都解析成功才核对答案要点，避免连带报错
            if not keys:
                E("有答案的题至少要有一个 answer_key")
            gold_text = norm(" ".join(units[(g["doc_id"], g["unit_id"])] for g in gold))
            for k in keys:
                if norm(k) not in gold_text:
                    E(f"答案要点不在 gold 原文中：「{k}」")
            for k in must_not:
                if norm(k) in gold_text:
                    E(f"版本陷阱「{k}」出现在当期 gold 原文里，陷阱不成立")
                other = [d for g in gold for d, _, _ in versions[g["family"]]
                         if d != g["doc_id"] and norm(k) in norm(units.get((d, g["unit_id"]), ""))]
                if not other:
                    warnings.append(f"{qid}：版本陷阱「{k}」在其他版本中也找不到，可能不是真陷阱")
        if q.get("type") == "version" and answerable and len(gold) == len(q["gold"]):
            # 真正的版本题：至少一条 gold 在其他版本里内容不同（或不存在）。否则换哪个版本答案都一样
            differs = any(norm(units.get((d, g["unit_id"]), "")) != norm(units[(g["doc_id"], g["unit_id"])])
                          for g in gold for d, _, _ in versions[g["family"]] if d != g["doc_id"])
            if not differs:
                E("标为版本题，但 gold 条款在各版本中内容相同")

        for g in gold:
            gold_split[(g["family"], g["unit_id"])].add((q.get("split"), qid))
        if q.get("group"):
            group_split[q["group"]].add((q.get("split"), qid))

        # gold 落到哪些检索块：被切开的长单元只算包含答案要点的那几块
        gold_chunks = []
        for g in gold:
            cands = chunks_of.get((g["doc_id"], g["unit_id"]), [])
            hit = [c for c, t in cands if any(norm(k) in norm(t) for k in keys)]
            gold_chunks.append(hit or [c for c, _ in cands])

        q_tok = tokens(q["question"]) if q.get("lang") == "en" else set()
        g_tok = set().union(*(tokens(units[(g["doc_id"], g["unit_id"])]) for g in gold)) if gold else set()
        overlap = round(len(q_tok & g_tok) / len(q_tok), 2) if q_tok and g_tok else None

        resolved.append({"id": qid, "type": q.get("type"), "pattern": q.get("pattern"), "lang": q.get("lang"),
                         "split": q.get("split"),
                         "question": q["question"], "as_of": str(as_of), "answerable": answerable,
                         "gold": gold, "gold_chunks": gold_chunks, "also_ok": also_ok, "near_miss": near_miss,
                         "answer_keys": keys, "must_not": must_not, "lexical_overlap": overlap})

    # ---- 切分纪律
    for (fam, uid), s in gold_split.items():
        if len({x[0] for x in s}) > 1:
            errors.append(f"条款 {fam}:{uid} 同时出现在开发集和测试集：{sorted(x[1] for x in s)}")
    for grp, s in group_split.items():
        if len({x[0] for x in s}) > 1:
            errors.append(f"group {grp} 的题分到了两侧：{sorted(x[1] for x in s)}")

    # ---- 统计
    print(f"评测集 {len(qs)} 题（默认调查日期 {default_as_of}）")
    tab = Counter((r["type"], r["split"]) for r in resolved)
    print(f"\n  {'题型':<14}{'dev':>5}{'test':>6}")
    for t in TYPES:
        print(f"  {t:<14}{tab[(t, 'dev')]:>5}{tab[(t, 'test')]:>6}")
    n_dev = sum(r["split"] == "dev" for r in resolved)
    print(f"  {'合计':<13}{n_dev:>5}{len(resolved) - n_dev:>6}")
    lang = Counter((r["lang"], r["split"]) for r in resolved)
    print(f"\n  语言：英文 {lang[('en','dev')]}/{lang[('en','test')]}，中文 {lang[('zh','dev')]}/{lang[('zh','test')]}（dev/test）")
    print(f"  无答案题：{sum(not r['answerable'] for r in resolved)}；"
          f"需要多条 gold 的题：{sum(len(r['gold']) > 1 for r in resolved)}")
    fam = Counter(g["family"] for r in resolved for g in r["gold"])
    print("  gold 来源：" + "，".join(f"{k} {v}" for k, v in fam.most_common()))
    ov = [r for r in resolved if r["lexical_overlap"] is not None]
    if ov:
        b = Counter("高 ≥0.6" if r["lexical_overlap"] >= 0.6 else "中 0.3–0.6" if r["lexical_overlap"] >= 0.3
                    else "低 <0.3" for r in ov)
        print(f"  英文题与 gold 原文的词面重合度：" + "，".join(f"{k} {b[k]}" for k in ("低 <0.3", "中 0.3–0.6", "高 ≥0.6")))
        top = sorted(ov, key=lambda r: -r["lexical_overlap"])[:5]
        print("    重合度最高：" + "，".join(f"{r['id']}({r['lexical_overlap']})" for r in top))
    if any(r.get("pattern") for r in resolved):
        pt = Counter((r["pattern"], r["split"]) for r in resolved)
        print("  按 pattern（dev/test）：" + "，".join(f"{p} {pt[(p, 'dev')]}/{pt[(p, 'test')]}" for p in PATTERNS
                                                    if pt[(p, "dev")] + pt[(p, "test")]))
    if golden.resolve() != GOLDEN.resolve() and GOLDEN.exists():
        # 与 v1 的重合：v1 的条款在设计 R6/R7 时被反复使用，v2 用到它们时结果可能偏乐观
        v1 = yaml.safe_load(open(GOLDEN, encoding="utf-8"))["questions"]
        v1_gold = {(g["family"], str(g["unit_id"])): q1["split"] for q1 in v1 for g in q1.get("gold") or []}
        hits = sorted({(g["family"], g["unit_id"], v1_gold[(g["family"], g["unit_id"])], r["id"])
                       for r in resolved for g in r["gold"] if (g["family"], g["unit_id"]) in v1_gold})
        print(f"  与 v1 gold 重合的条款：{len(hits)} 处" + ("" if not hits else "：" + "，".join(
            f"{r} {f}:{u}（v1 {s}）" for f, u, s, r in hits)))
    multi_chunk = [r["id"] for r in resolved if any(len(c) > 1 for c in r["gold_chunks"])]
    if multi_chunk:
        print(f"  gold 单元被切成多块、且多块都含答案要点的题：{multi_chunk}")

    if warnings:
        print("\n提示：")
        for w in warnings:
            print(f"  · {w}")
    if errors:
        print(f"\n校验未通过（{len(errors)} 项）：")
        for e in errors:
            print(f"  ✗ {e}")
        raise SystemExit(1)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resolved, f, ensure_ascii=False, indent=1)
    print(f"\n校验通过：全部答案要点在调查日期有效的原文中逐字找到；已写出 {out_path}")


if __name__ == "__main__":
    main()