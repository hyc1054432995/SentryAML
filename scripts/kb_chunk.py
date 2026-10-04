"""
SentryAML - RAG 层 R2d：从结构单元生成检索块（chunk）
====================================================
结构单元（unit）是"引用"的最小单位；检索块（chunk）是"向量化和召回"的单位。两者大多一一对应，
但有两种情况需要调整：

1. 太长（> MAX_CHARS）：Guidelines 附录 A 有 4,800 字，一个向量装不下这么多意思，
   而且会把相关的一小段淹没在无关内容里。
   → 在列表项边界切开，每一块都带上引导句（lead-in），例如
     "Examples of potentially higher risk categories under paragraph 8.7 of the Notice include ―"
     否则切出来的 "(b) Country risk ..." 单独看不知道在说什么。
   → 没有列表结构的长段，按句子切。

2. 太短（< MIN_CHARS）：CEF 附录 A 的风险信号一条只有一两百字，Notice 里也有 "7.2 The assessment
   of low risks shall be supported by ..." 这样的一句话段落。向量太短，语义不稳定。
   → 与同一标题下紧邻的短段合并，但块里记录包含的全部 unit_id，引用时仍然能落到具体条款。
   → 定义条目、方框不合并：定义要精确对应术语，案例要完整。

另外：
  - 脚注附在引用它的块后面（"[^14] This refers to the point in time when ..."）——
    脚注常常承载关键定义，例如 "suspicion was first established" 指什么时点。
  - 标题不单独成块，它作为 heading_path 保存，R5 做上下文增强时拼到块前面。

用法：
    python scripts/kb_chunk.py           # 全部文件，输出 data/corpus/chunks/<doc_id>.jsonl 并打印统计
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSED = PROJECT_ROOT / "data" / "corpus" / "parsed"
OUT = PROJECT_ROOT / "data" / "corpus" / "chunks"

MAX_CHARS = 1800        # 约 450 token
MIN_CHARS = 300         # 约 75 token
MERGE_UP_TO = 1200

CHUNK_KINDS = {"paragraph", "definition", "block", "box", "endnote"}
NO_MERGE = {"definition", "box", "endnote"}
ENUM_LINE = re.compile(r"^(\(?[a-z]{1,2}\)|\([ivxlc]+\)|[ivxlc]+\)|•|o\s)")


def _split_sentences(text: str, limit: int) -> list[str]:
    parts, buf = [], ""
    for s in re.split(r"(?<=[.;])\s+(?=[A-Z(“\"])", text):
        if buf and len(buf) + len(s) + 1 > limit:
            parts.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        parts.append(buf)
    return parts


def _hard_wrap(text: str, limit: int) -> list[str]:
    """最后的保险：连句号都没有的超长文本（例如表格转写），按词切。"""
    out, buf = [], ""
    for w in text.split(" "):
        if buf and len(buf) + len(w) + 1 > limit:
            out.append(buf)
            buf = w
        else:
            buf = f"{buf} {w}".strip()
    return out + ([buf] if buf else [])


def split_long(text: str) -> list[str]:
    """超长单元：优先按列表项切，每块带上引导句；否则按句子切。"""
    lines = text.split("\n")
    if len(lines) > 1:
        # 第一行本身就是列表项（附录 B 的 "i) ..."、附录 A 的表格行）→ 没有引导句
        lead, items = ("", lines) if ENUM_LINE.match(lines[0]) else (lines[0], lines[1:])
        room = MAX_CHARS - len(lead) - 1
        if len(lead) < MAX_CHARS * 0.5:
            groups, buf = [], []
            for it in items:
                pieces = [it] if len(it) <= room else [
                    q for p in _split_sentences(it, room) for q in (_hard_wrap(p, room) if len(p) > room else [p])]
                for p in pieces:
                    if buf and sum(len(x) + 1 for x in buf) + len(p) > room:
                        groups.append(buf)
                        buf = []
                    buf.append(p)
            if buf:
                groups.append(buf)
            return [(lead + "\n" if lead else "") + "\n".join(g) for g in groups]
    parts = _split_sentences(text.replace("\n", " "), MAX_CHARS)
    return [q for p in parts for q in (_hard_wrap(p, MAX_CHARS) if len(p) > MAX_CHARS else [p])]


def build_chunks(doc_id: str, units: list[dict]) -> list[dict]:
    fns = {u["unit_id"]: u for u in units if u["kind"] == "footnote"}
    body = [u for u in units if u["kind"] in CHUNK_KINDS and u["text"]]

    def make(us: list[dict], text: str, suffix: str = "") -> dict:
        refs = sorted({n for u in us for n in u.get("footnote_refs", [])})
        notes = [f"[^{n}] {fns[f'fn{n}']['text']}" for n in refs if f"fn{n}" in fns]
        full = text + ("\n\n" + "\n".join(notes) if notes else "")
        return {
            "chunk_id": f"{doc_id}::{us[0]['unit_id']}{suffix}",
            "doc_id": doc_id,
            # 附在块末尾的脚注也算块的组成单元：脚注不单独成块，只能经由引用它的块被检索到，
            # 不登记进来的话，"这个块里有 fn17"就查不到，引用和评测都会漏掉脚注（R4 发现）
            "unit_ids": [u["unit_id"] for u in us] + [f"fn{n}" for n in refs if f"fn{n}" in fns],
            "kind": us[0]["kind"],
            "heading_path": us[0]["heading_path"],
            "text": full,
            "page_start": min(u["page_start"] for u in us),
            "page_end": max(u["page_end"] for u in us),
            "amended_by": sorted({a for u in us for a in u.get("amended_by", [])}),
            "n_chars": len(full),
        }

    chunks, i = [], 0
    while i < len(body):
        u = body[i]
        if len(u["text"]) > MAX_CHARS:
            parts = split_long(u["text"])
            for k, p in enumerate(parts, 1):
                chunks.append(make([u], p, f"#{k}" if len(parts) > 1 else ""))
            i += 1
            continue
        group = [u]
        if u["kind"] not in NO_MERGE:
            while len("\n".join(g["text"] for g in group)) < MIN_CHARS and i + 1 < len(body):
                nxt = body[i + 1]
                total = len("\n".join(g["text"] for g in group)) + len(nxt["text"])
                if (nxt["kind"] in NO_MERGE or nxt["heading_path"] != u["heading_path"]
                        or total > MERGE_UP_TO):
                    break
                group.append(nxt)
                i += 1
        chunks.append(make(group, "\n".join(g["text"] for g in group)))
        i += 1
    return chunks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_lens = []
    print(f"{'文件':<26}{'单元':>6}{'块':>6}{'拆分':>6}{'合并':>6}{'中位字数':>9}{'最长':>7}")
    for f in sorted(PARSED.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            units = [json.loads(l) for l in fh]
        chunks = build_chunks(f.stem, units)
        with open(OUT / f"{f.stem}.jsonl", "w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        lens = sorted(c["n_chars"] for c in chunks)
        all_lens += lens
        n_split = sum(1 for c in chunks if "#" in c["chunk_id"])
        n_merged = sum(1 for c in chunks if len(c["unit_ids"]) > 1)
        n_units = sum(1 for u in units if u["kind"] in CHUNK_KINDS)
        print(f"{f.stem:<26}{n_units:>6}{len(chunks):>6}{n_split:>6}{n_merged:>6}"
              f"{lens[len(lens) // 2]:>9}{lens[-1]:>7}")
    all_lens.sort()
    n = len(all_lens)
    over = sum(1 for x in all_lens if x > MAX_CHARS)
    print(f"\n共 {n} 块；字数 P10 {all_lens[n // 10]}，中位 {all_lens[n // 2]}，P90 {all_lens[n * 9 // 10]}，"
          f"最长 {all_lens[-1]}；超过 {MAX_CHARS} 字的 {over} 块（超出部分是附在后面的脚注）")


if __name__ == "__main__":
    main()