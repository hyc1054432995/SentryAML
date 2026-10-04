"""
SentryAML - RAG 层 R2：版本对比（同时也是解析器的交叉验证）
==========================================================
把同一份文件的两个版本按条款编号对齐，找出新增、删除、改动的条款。

两个用途：
  1. 业务：Agent 引用条款时可以说明"该条款在 2025 年修订中有改动"；合规人员也能一眼看到修订范围。
  2. 验证解析器：MAS 在修订版里给改动过的条款打了标记 [MAS Notice 626 (Amendment) 2025]。
     这些标记是 MAS 自己写的"标准答案"。如果解析器切得对，
     "文本对比发现的改动"应当和"MAS 标记的改动"高度重合。
     两者对不上的地方，要么是解析器切错了，要么是标记挂错了单元——都值得人工看一眼。

用法：
    python scripts/kb_version_diff.py mas_n626_2024 mas_n626_2025
    python scripts/kb_version_diff.py mas_n626_guidelines_2024 mas_n626_guidelines_2025
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSED = PROJECT_ROOT / "data" / "corpus" / "parsed"
CONTENT_KINDS = {"paragraph", "definition", "block", "box", "footnote"}


def load(doc_id: str) -> dict[str, dict]:
    with open(PARSED / f"{doc_id}.jsonl", encoding="utf-8") as f:
        units = [json.loads(l) for l in f]
    return {u["unit_id"]: u for u in units if u["kind"] in CONTENT_KINDS}


def norm(t: str) -> str:
    """比较前归一：引号、空白、脚注引用编号（修订常常只是脚注重新编号）。"""
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"\[\^\d+\]", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def main():
    old_id, new_id = sys.argv[1], sys.argv[2]
    old, new = load(old_id), load(new_id)
    # 脚注按编号对齐没有意义：新版插入一个脚注，后面的全部顺延。改为按内容对齐。
    fn_old = {norm(u["text"]) for k, u in old.items() if u["kind"] == "footnote"}
    fn_new = {norm(u["text"]): k for k, u in new.items() if u["kind"] == "footnote"}
    fn_added = sorted((fn_new[t] for t in set(fn_new) - fn_old), key=lambda x: int(x[2:]))
    fn_removed = len(fn_old - set(fn_new))
    old = {k: u for k, u in old.items() if u["kind"] != "footnote"}
    new_fn = {k: new[k] for k in fn_added}
    new = {k: u for k, u in new.items() if u["kind"] != "footnote"}
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(k for k in set(old) & set(new) if norm(old[k]["text"]) != norm(new[k]["text"]))
    same = len(set(old) & set(new)) - len(changed)

    print(f"{old_id} → {new_id}")
    print(f"  条款单元：旧 {len(old)}，新 {len(new)}；未变 {same}，改动 {len(changed)}，"
          f"新增 {len(added)}，删除 {len(removed)}")
    for title, ids in (("新增", added), ("删除", removed), ("改动", changed)):
        if ids:
            print(f"\n  {title}：{', '.join(ids[:40])}{' …' if len(ids) > 40 else ''}")
    print(f"\n  脚注（按内容对齐，编号顺延不算改动）：新增或改写 {len(fn_added)} 条 "
          f"{fn_added}，删除或改写 {fn_removed} 条")

    tagged = {k for k, u in {**new, **new_fn}.items() if u["amended_by"]}
    if tagged:
        diff = set(changed) | set(added) | set(fn_added)
        hit = tagged & diff
        print(f"\n  与 MAS 修订标记交叉核对：")
        print(f"    MAS 标记的单元 {len(tagged)} 个，其中文本确有改动或为新增 {len(hit)} 个"
              f"（{len(hit) / len(tagged):.0%}）")
        print(f"    文本有改动/新增的单元 {len(diff)} 个，其中带标记 {len(hit)} 个"
              f"（{len(hit) / max(len(diff), 1):.0%}）")
        miss_tag = sorted(diff - tagged)
        tag_nodiff = sorted(tagged - diff)
        if tag_nodiff:
            print(f"    有标记但文本没变（标记可能挂错了单元，或只是格式调整）：{', '.join(tag_nodiff)}")
        if miss_tag:
            print(f"    文本变了但没有标记（可能是解析差异，需人工看）：{', '.join(miss_tag[:30])}")

    if len(sys.argv) > 3:                                  # 查看某一条的前后对比
        k = sys.argv[3]
        for label, d in (("旧", old), ("新", new)):
            print(f"\n  [{label}] {k}：{d.get(k, {}).get('text', '（不存在）')}")


if __name__ == "__main__":
    main()