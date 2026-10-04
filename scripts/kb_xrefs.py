"""
SentryAML - RAG 层 R2d：交叉引用抽取
====================================
监管文本里到处是 "as required by paragraph 6"、"Paragraphs 6.19-6.26 of MAS Notice 626"、
"see Box 1" 这类引用。把它们抽成"条款 → 条款"的边，检索时就能沿着引用补全上下文：
检索命中 TM 指引的 2.10 段，而这段说"参见 Notice 6.19–6.26"，就把那几段一起带上。

这是 D39 里说的"用规则确定地解析出来的图"：边来自原文的明文引用，不靠大模型猜。

三类边：
  internal   同一份文件内部：Notice 4.2 → 4.1；FATF 某段 → box-1
  cross_doc  跨文件：TM 指引 → MAS Notice 626 第 6.19 段；Guidelines → Notice
  external   引用法律条文：section 2 of the Banking Act —— 不在知识库里，只记录，不解析

目标只记录"文件族 + 条款编号"，不绑定具体版本：
  查询时再按调查日期选出当时有效的那个版本（kb_documents_as_of），和检索保持同一套版本逻辑。

用法：
    python scripts/kb_xrefs.py            # 抽取全部文件，打印解析率与未解析样例
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSED = PROJECT_ROOT / "data" / "corpus" / "parsed"

FAMILY_OF = [("mas_n626_guidelines", "mas_n626_guidelines"), ("mas_n626", "mas_n626"),
             ("mas_tm_guidance", "mas_tm_guidance")]

# 条款编号的三种写法（顺序重要：先试点号、再试连字符、最后纯数字）
ID_DOT = r"\d{1,2}[A-Z]?\.\d{1,2}[A-Z]?"                 # 6.19 / 6.11A / 11A.3
ID_HYPHEN = r"\d{1,2}(?:-(?:\d{1,2}[A-Z]?|[A-Z])){1,3}"   # 6-10-10 / 14-A
ID_PLAIN = r"\d{1,2}[A-Z]?"                              # 6（整章）
ID_RE = re.compile(rf"({ID_DOT}|{ID_HYPHEN}|{ID_PLAIN})((?:\([a-z0-9]{{1,4}}\))*)")
SEP_RE = re.compile(r"\s*(,\s*(?:and|or)?|and|or|to|–|-)\s*")
PARA_KW = re.compile(r"\b(sub)?paragraphs?\s+", re.I)
TARGET_NOTICE = re.compile(r"^\s*(?:of|in|under)\s+(?:the\s+)?(?:MAS\s+)?Notice(?:\s+626)?\b", re.I)
TARGET_GUIDE = re.compile(r"^\s*(?:of|in|under)\s+(?:these|the)\s+Guidelines\b|^\s*of\s+the\s+Guidelines\s+to\s+MAS\s+Notice", re.I)
LABEL_RE = re.compile(r"\b(Box|Annex|Appendix|APPENDIX|Section|SECTION)\s+([A-Z]|\d{1,2}|[IVX]{1,5})\b")
STATUTE_RE = re.compile(r"\b[Ss]ections?\s+(\d{1,3}[A-Z]{0,2}(?:\(\d+\))*)\s+of\s+(?:the\s+)?"
                        r"((?:[A-Z][\w()]*\s+){0,8}?Act(?:\s+\d{4})?|[A-Z]{2,6}(?:\s+Act)?)")


_FAMILY_MAP: dict[str, str] = {}          # 由 kb_load.py 从 kb_documents 读入；单独运行时用下面的前缀规则


def family(doc_id: str) -> str:
    if doc_id in _FAMILY_MAP:
        return _FAMILY_MAP[doc_id]
    for prefix, fam in FAMILY_OF:
        if doc_id.startswith(prefix):
            return fam
    return doc_id


def load_units() -> dict[str, list[dict]]:
    out = {}
    for f in sorted(PARSED.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            out[f.stem] = [json.loads(l) for l in fh]
    return out


def _parse_id_list(text: str, pos: int) -> tuple[list[tuple[str, str]], int]:
    """从 pos 开始读 "6.19, 6.20 and 6.21" / "6.19-6.26" / "6, 7 and 8"，返回 [(编号, 连接符)] 和结束位置。"""
    items, i, sep = [], pos, ""
    while True:
        m = ID_RE.match(text, i)
        if not m:
            break
        items.append((m.group(1), sep))
        i = m.end()
        s = SEP_RE.match(text, i)
        if not s or not ID_RE.match(text, s.end()):
            break
        sep = s.group(1).strip()
        i = s.end()
    return items, i


def _expand(items: list[tuple[str, str]], order: list[str]) -> list[str]:
    """把 "6.19 - 6.26"、"6-10-10 to 6-10-12" 这样的区间，按目标文件里的条款顺序展开。"""
    out = []
    for k, (uid, sep) in enumerate(items):
        if sep in ("to", "–", "-") and out and out[-1] in order and uid in order:
            a, b = order.index(out[-1]), order.index(uid)
            out.extend(order[a + 1:b + 1])
        else:
            out.append(uid)
    return out


def extract(all_units: dict[str, list[dict]], family_map: dict[str, str] | None = None) -> list[dict]:
    if family_map:
        _FAMILY_MAP.update(family_map)
    # 每个文件族：条款编号的顺序（用最新版本），以及全部版本里出现过的编号
    fam_order: dict[str, list[str]] = {}
    fam_ids: dict[str, set[str]] = defaultdict(set)
    fam_headings: dict[str, dict[str, str]] = defaultdict(dict)
    for doc_id, units in sorted(all_units.items()):
        fam = family(doc_id)
        fam_order[fam] = [u["unit_id"] for u in units if u["kind"] != "heading"]   # 后面的版本覆盖前面的
        for u in units:
            fam_ids[fam].add(u["unit_id"])
            if u["kind"] == "heading":
                fam_headings[fam][u["text"].upper()] = u["unit_id"]

    def heading_for(fam: str, label: str) -> str | None:
        lab = label.upper()
        for text, uid in fam_headings[fam].items():
            if text == lab or text.startswith(lab + " ") or text.startswith(lab + ":") or text.startswith(lab + " –"):
                return uid
        return None

    edges = []
    for doc_id, units in all_units.items():
        src_fam = family(doc_id)
        for u in units:
            if u["kind"] == "heading":
                continue
            text = u["text"]
            base = dict(src_doc_id=doc_id, src_unit_id=u["unit_id"])

            # ---- 段落引用
            for kw in PARA_KW.finditer(text):
                if kw.group(1):                                  # "subparagraph (a)" 只是本段内的小项
                    continue
                items, end = _parse_id_list(text, kw.end())
                if not items:
                    continue
                tail = text[end:end + 60]
                if TARGET_NOTICE.match(tail):
                    tgt, kind = "mas_n626", "cross_doc"
                elif TARGET_GUIDE.match(tail):
                    tgt, kind = "mas_n626_guidelines", "cross_doc"
                else:
                    tgt, kind = src_fam, "internal"
                if tgt == src_fam:
                    kind = "internal"
                raw = text[kw.start():end + (len(TARGET_NOTICE.match(tail).group(0)) if TARGET_NOTICE.match(tail) else 0)]
                for uid in _expand(items, fam_order.get(tgt, [])):
                    t_fam, t_kind = tgt, kind
                    # 编号语法本身就指明了文件：Guidelines 自己的编号用连字符（6-10-10），
                    # 它提到的点号编号（6.3(b)）只可能是 Notice 的条款
                    if tgt == "mas_n626_guidelines" and re.fullmatch(ID_DOT, uid):
                        t_fam, t_kind = "mas_n626", "cross_doc"
                    target = uid if uid in fam_ids[t_fam] else (f"§{uid}" if f"§{uid}" in fam_ids[t_fam] else uid)
                    edges.append({**base, "raw": raw.strip()[:200], "kind": t_kind, "target_family": t_fam,
                                  "target_unit_id": target, "resolved": target in fam_ids[t_fam]})

            # ---- Box / Annex / Appendix / Section（同一文件内）
            for m in LABEL_RE.finditer(text):
                word, lab = m.group(1), m.group(2)
                if word.lower() == "section" and lab.isdigit() and \
                        (word == "section" or text[m.end():m.end() + 4] == " of "):
                    continue                                     # "section 16 of the ... Act" 是法律条文，下面单独处理
                if word.lower() == "box":
                    target = f"box-{lab}"
                    ok = target in fam_ids[src_fam]
                else:
                    target = (heading_for(src_fam, f"{word} {lab}")
                              or (f"§{lab}" if f"§{lab}" in fam_ids[src_fam] else None)   # "Section 3" → 标题 "3 ..."
                              or f"{word} {lab}")
                    ok = target in fam_ids[src_fam]
                if target == u["unit_id"] or (u["kind"] == "box" and text.startswith(f"Box {lab}")):
                    continue                                     # 方框自己的标题
                edges.append({**base, "raw": m.group(0), "kind": "internal", "target_family": src_fam,
                              "target_unit_id": target, "resolved": ok})

            # ---- 法律条文（外部，只记录）
            for m in STATUTE_RE.finditer(text):
                edges.append({**base, "raw": m.group(0)[:200], "kind": "external",
                              "target_family": re.sub(r"\s+", " ", m.group(2)).strip(),
                              "target_unit_id": f"s{m.group(1)}", "resolved": False})

    # 去重（同一段里重复提到同一目标）
    seen, uniq = set(), []
    for e in edges:
        k = (e["src_doc_id"], e["src_unit_id"], e["kind"], e["target_family"], e["target_unit_id"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq


def report(edges: list[dict]) -> None:
    by = Counter((e["kind"], e["resolved"]) for e in edges)
    print(f"交叉引用共 {len(edges)} 条")
    for kind in ("internal", "cross_doc", "external"):
        ok, bad = by[(kind, True)], by[(kind, False)]
        if kind == "external":
            print(f"  {kind:<10}{ok + bad:>6} 条（法律条文，不在知识库内，只记录）")
        else:
            print(f"  {kind:<10}{ok + bad:>6} 条，解析成功 {ok}（{ok / max(ok + bad, 1):.0%}）")
    per_doc = Counter(e["src_doc_id"] for e in edges if e["kind"] != "external")
    print("  按来源文件：" + "，".join(f"{d} {n}" for d, n in sorted(per_doc.items())))
    acts = Counter(e["target_family"] for e in edges if e["kind"] == "external")
    print("  最常引用的法律：" + "，".join(f"{a}({n})" for a, n in acts.most_common(5)))
    bad = [e for e in edges if not e["resolved"] and e["kind"] != "external"]
    if bad:
        print(f"\n  未解析样例（{len(bad)} 条，最多显示 15 条）：")
        for e in bad[:15]:
            print(f"    {e['src_doc_id']}:{e['src_unit_id']} → {e['target_family']}:{e['target_unit_id']}  「{e['raw'][:70]}」")


if __name__ == "__main__":
    edges = extract(load_units())
    report(edges)
    cross = [e for e in edges if e["kind"] == "cross_doc"][:8]
    print("\n  跨文件引用样例：")
    for e in cross:
        print(f"    {e['src_doc_id']}:{e['src_unit_id']} → {e['target_family']}:{e['target_unit_id']}  「{e['raw'][:70]}」")