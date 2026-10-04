"""
SentryAML - RAG 层 R2b：按条款解析（MAS 文件）
==============================================
输入：kb_layout.extract_layout() 给出的干净文本行
输出：结构单元（unit）列表，每个单元是一个可以被精确引用的最小法律单位：
        段落 14.1、Guidelines 的 14-1、定义条目 2.1 "beneficial owner"、脚注、修订尾注……

为什么按条款切，而不是按固定长度切：
  - 引用必须落到条款编号（"MAS Notice 626 第 14.2 段"），固定长度的块会把两段话切在一起，
    或把一段话切成两半，引用就对不上了。
  - 条款是法律文本天然的语义单位：一段只规定一件事，检索粒度正好。
  - 每个单元带着它的"标题路径"（14 SUSPICIOUS TRANSACTIONS REPORTING > 14.2），
    后面做上下文增强、父子扩展都靠它。

非 MAS 文件（FATF、STR 指南、网页、论文）没有统一编号，交给 kb_parser_generic.py 按排版信号切分。

规则全部来自对真实排版的观察，写在 PROFILES 里；每类文件一套：
  mas_notice      段落编号 14.1 / 6.11A，章标题 "14 SUSPICIOUS ..."（粗体）
  mas_guidelines  段落编号 14-1 / 1-4-13 / 14-A / B-1-1，标题 "1-4 Key Concepts"（粗体）
  mas_tm          段落编号 2.11，章标题字号大于正文（不加粗）

用法：
    python scripts/kb_parser.py                  # 解析 data/corpus/raw 下的全部文件
    python scripts/kb_parser.py mas_n626_2025    # 只解析一份
输出：data/corpus/parsed/<doc_id>.jsonl，并打印质量报告
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kb_layout import DocLayout, Line, extract_layout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "corpus" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "corpus" / "parsed"

# ---------------------------------------------------------------------------
# 每类文件的解析规则
# ---------------------------------------------------------------------------
PROFILES = {
    "mas_notice": dict(
        para_re=r"^(\d{1,2}[A-Z]?\.\d{1,2}[A-Z]?)(?=\s|$)",           # 14.1 / 6.11A / 11A.3
        heading_re=r"^(\d{1,2}[A-Z]?)\s+(\S.*)$",                        # 14 ... / 11A ...
        heading_needs_bold=True,
        def_para="2.1",                                           # 定义集中在 2.1
        amend_re=r"^\[MAS Notice 626 \(Amendment\) \d{4}\]$",
        endnote_heading="Endnotes on History of Amendments",
        appendix_re=r"^Appendix \d+$",
    ),
    "mas_guidelines": dict(
        para_re=r"^((?:\d{1,2}|[A-Z])(?:-(?:\d{1,2}[A-Z]?|[A-Z])){1,3})(?=\s|$)",
        heading_re=r"^((?:\d{1,2}|[A-Z])(?:-\d{1,2}){0,2})\s+(\S.*)$",
        heading_needs_bold=True,
        def_para=None,
        amend_re=r"^\[.*Amendment.*\]$",
        endnote_heading=None,
        appendix_re=r"^APPENDIX [A-Z]\b.*$",
    ),
    "mas_tm": dict(
        para_re=r"^(\d{1,2}\.\d{1,2})(?=\s|$)",
        heading_re=r"^(\d{1,2})\s+(\S.*)$",
        heading_needs_bold=False,                                 # 章标题靠字号识别
        def_para=None,
        amend_re=None,
        endnote_heading=None,
        appendix_re=r"^Annex [A-Z0-9]+.*$",
        box_re=r"^(Box Story:.*|Best Practices|Case Study [A-Z])$",   # 方框里的案例与良好实践
    ),
}
PROFILE_OF = {                                                    # doc_id 前缀 → 规则
    "mas_n626_guidelines": "mas_guidelines",
    "mas_n626": "mas_notice",
    "mas_tm_guidance": "mas_tm",
}

ENUM_RE = re.compile(r"^(\(?[a-z]{1,2}\)|\([ivxlc]+\)|[ivxlc]+\)|\([IVXLC]+\)|•|o\s)")
ROMAN_SUBHEAD_RE = re.compile(r"^(?:\(?[IVX]+\)|[ivx]+\.)\s+\S")  # "(III) Identification ..." / "ii. Back-testing"


@dataclass
class Unit:
    doc_id: str
    unit_id: str
    kind: str                        # front | heading | paragraph | definition | block | box | footnote | endnote
    heading_path: list[str]
    text: str
    page_start: int
    page_end: int
    parent_id: str | None = None
    footnote_refs: list[int] = field(default_factory=list)
    amended_by: list[str] = field(default_factory=list)
    order: int = 0


def profile_for(doc_id: str) -> str | None:
    for prefix, prof in PROFILE_OF.items():
        if doc_id.startswith(prefix):
            return prof
    return None


def _id_key(uid: str) -> tuple:
    """段落编号的主号，用来拒绝"倒退"的误识别（例如正文换行后行首恰好是 '15.7,'）。
    11A 排在 11 之后、12 之前；附录编号（B-1-1）单独一类，不参与比较。"""
    first = re.split(r"[.\-]", uid)[0]
    m = re.fullmatch(r"(\d+)([A-Z]?)", first)
    return (0, int(m.group(1)), m.group(2)) if m else (1, 0, first)


def _unique(base: str, units: list) -> str:
    used = {u.unit_id for u in units}
    if base not in used:
        return base
    k = 2
    while f"{base}#{k}" in used:
        k += 1
    return f"{base}#{k}"


def _join(prev: str, line: str) -> str:
    if not prev:
        return line
    if ENUM_RE.match(line):
        return prev + "\n" + line
    if prev.endswith("-") and line[:1].islower():
        return prev + line
    return prev + " " + line


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def parse(doc_id: str, lay: DocLayout, prof_name: str) -> list[Unit]:
    P = PROFILES[prof_name]
    para_re = re.compile(P["para_re"])
    head_re = re.compile(P["heading_re"])
    amend_re = re.compile(P["amend_re"]) if P["amend_re"] else None
    appx_re = re.compile(P["appendix_re"]) if P["appendix_re"] else None
    body = lay.body_size

    units: list[Unit] = []
    heads: list[tuple[int, str, str]] = []        # (层级, 编号, 标题)
    cur: Unit | None = None
    seen_ids: set[str] = set()
    major = (0, 0, "")
    in_endnotes = False
    last_heading_line: Line | None = None

    def path() -> list[str]:
        return [f"{hid} {t}".strip() for _, hid, t in heads]

    def open_unit(uid: str, kind: str, text: str, ln: Line, parent: str | None = None) -> Unit:
        nonlocal cur
        u = Unit(doc_id, uid, kind, path(), text, ln.page, ln.page, parent, list(ln.footnote_refs))
        units.append(u)
        cur = u
        return u

    def push_heading(level: int, hid: str, title: str, ln: Line):
        nonlocal cur
        while heads and heads[-1][0] >= level:
            heads.pop()
        heads.append((level, hid, title))
        uid = hid or f"H{len(units)}"
        units.append(Unit(doc_id, _unique(f"§{uid}", units), "heading", path()[:-1],
                          f"{hid} {title}".strip(), ln.page, ln.page,
                          heads[-2][1] if len(heads) > 1 else None, list(ln.footnote_refs)))
        cur = None

    def numbered_depth() -> int:
        return max((lv for lv, hid, _ in heads if hid), default=0)

    for ln in lay.lines:
        t = ln.text.strip()
        is_bigger = ln.size >= body + 0.9
        looks_heading = (ln.bold or is_bigger) and len(t) < 140 and not t.endswith((";", ","))

        # ---- 修订标记：挂到当前单元上，不进入正文
        if amend_re and amend_re.match(t):
            if cur is not None:
                cur.amended_by.append(t.strip("[]"))
            continue

        # ---- 附录标题（放在尾注判断之前：附录出现即结束尾注区）
        if appx_re and appx_re.match(t) and (ln.bold or is_bigger):
            in_endnotes = False
            if "–" in t:
                push_heading(1, t.split("–")[0].strip(), t.split("–", 1)[1].strip(), ln)
            else:
                push_heading(1, t, "", ln)
            last_heading_line = ln
            continue

        # ---- 修订历史尾注
        if P["endnote_heading"] and t.startswith(P["endnote_heading"]):
            in_endnotes = True
            push_heading(1, "", t, ln)
            continue
        if in_endnotes:
            m = re.match(r"^(\d+)\.\s*(.*)$", t)
            if m:
                open_unit(f"endnote-{m.group(1)}", "endnote", m.group(2), ln)
            elif cur is not None:
                cur.text = _join(cur.text, t)
                cur.page_end = ln.page
            continue

        # ---- 带编号的标题（"14 SUSPICIOUS ..." / "1-4 Key Concepts" / "B-2 ..."）
        m = head_re.match(t)
        styled = ln.bold if P["heading_needs_bold"] else (ln.bold or is_bigger)
        if m and styled and (ln.bold or not para_re.match(t)):
            hid, title = m.group(1), m.group(2)
            level = hid.count("-") + hid.count(".") + 1
            push_heading(level, hid, title, ln)
            last_heading_line = ln
            continue

        # ---- 段落编号
        m = para_re.match(t)
        if m and not ln.bold:
            uid = m.group(1)
            k = _id_key(uid)
            if uid not in seen_ids and (k >= major or k[0] == 1):
                seen_ids.add(uid)
                major = max(major, k) if k[0] == 0 else major
                open_unit(uid, "paragraph", t[m.end():].strip(), ln)
                last_heading_line = None
                continue

        # ---- 不带编号的小标题（"Risk Assessment" / "Governance" / "(III) Identification ..."）
        if (ROMAN_SUBHEAD_RE.match(t) or (looks_heading and not ENUM_RE.match(t))) and len(t) < 140:
            # 紧接上一个标题行、且同样是标题样式 → 标题换行，合并
            if last_heading_line is not None and ln.page == last_heading_line.page \
                    and ln.y0 - last_heading_line.y1 < ln.size * 0.9 and heads:
                lv, hid, title = heads[-1]
                heads[-1] = (lv, hid, title + " " + t)
                units_head = [u for u in units if u.kind == "heading"][-1]
                units_head.text = (units_head.text + " " + t).strip()
                last_heading_line = ln
                continue
            push_heading(numbered_depth() + 1, "", t, ln)
            last_heading_line = ln
            continue

        last_heading_line = None

        # ---- 定义条目（Notice 2.1 里以引号开头的行）
        if P["def_para"] and cur is not None and (cur.unit_id == P["def_para"] or cur.kind == "definition") \
                and t[:1] in "“\"":
            term = re.match(r"^[“\"]([^”\"]+)[”\"]", t)
            name = term.group(1) if term else t[:30]
            open_unit(f"{P['def_para']}:{name}", "definition", t, ln, parent=P["def_para"])
            continue

        # ---- 方框（案例、良好实践）：单独成块，不并入前一段
        if P.get("box_re") and re.match(P["box_re"], t):
            base = re.sub(r"[^A-Za-z0-9]+", "-", t.split(":")[0]).strip("-").lower()
            base = "case-" + t.split()[-1] if t.startswith("Case Study") else f"{base}-p{ln.page}"
            open_unit(_unique(base, units), "box", t, ln, parent=cur.unit_id if cur else None)
            continue

        # ---- 续行
        if cur is None:
            if heads:
                hid = heads[-1][1] or re.sub(r"[^A-Za-z0-9]+", "-", heads[-1][2]).strip("-").lower()[:40]
                open_unit(_unique(hid, units), "block", t, ln)
            else:
                open_unit(_unique("front", units), "front", t, ln)
            continue
        cur.text = _join(cur.text, t)
        cur.page_end = ln.page
        cur.footnote_refs.extend(ln.footnote_refs)

    # ---- 脚注：挂到第一次引用它的单元下
    ref_owner = {}
    for u in units:
        for n in u.footnote_refs:
            ref_owner.setdefault(n, u)
    for fn in lay.footnotes:
        owner = ref_owner.get(fn.num)
        text, amended = fn.text, []
        if amend_re:
            for tag in re.findall(r"\[[^\]]*Amendment[^\]]*\]", text):
                amended.append(tag.strip("[]"))
                text = text.replace(tag, "").strip()
        units.append(Unit(doc_id, f"fn{fn.num}", "footnote",
                          owner.heading_path + [owner.unit_id] if owner else [],
                          text, fn.page, fn.page, owner.unit_id if owner else None,
                          [], amended))

    for i, u in enumerate(units):
        u.order = i
        u.text = u.text.strip()
        u.footnote_refs = sorted(set(u.footnote_refs))
    return units


# ---------------------------------------------------------------------------
# 质量报告
# ---------------------------------------------------------------------------
def quality_report(units: list[Unit], lay: DocLayout) -> dict:
    by_kind = {}
    for u in units:
        by_kind[u.kind] = by_kind.get(u.kind, 0) + 1

    # 编号连续性：同一主号下，数字部分应当连续（6.1, 6.2, ... 允许 6.11A 这类插入）
    gaps = []
    groups: dict[str, list[int]] = {}
    for u in units:
        if u.kind != "paragraph":
            continue
        m = re.match(r"^(\w+)[.\-](\d+)$", u.unit_id)
        if m:
            groups.setdefault(m.group(1), []).append(int(m.group(2)))
    plain = sorted(int(u.unit_id) for u in units if u.kind == "paragraph" and u.unit_id.isdigit())
    if plain:
        groups["(全局编号)"] = plain
    for g, nums in groups.items():
        s = sorted(set(nums))
        missing = sorted(set(range(s[0], s[-1] + 1)) - set(s))
        if missing:
            gaps.append(f"{g}: 缺 {missing}")

    refs = {n for u in units for n in u.footnote_refs}
    fns = {int(u.unit_id[2:]) for u in units if u.kind == "footnote"}
    orphan_fn = sorted(fns - refs)
    missing_fn = sorted(refs - fns)
    lens = sorted(len(u.text) for u in units if u.kind in ("paragraph", "definition", "block", "box"))
    dup = [k for k, v in __import__("collections").Counter(u.unit_id for u in units).items() if v > 1]
    return {
        "units": by_kind,
        "numbering_gaps": gaps,
        "footnotes_unreferenced": orphan_fn,
        "footnote_refs_without_text": missing_fn,
        "amended_units": sum(1 for u in units if u.amended_by),
        "duplicate_ids": dup,
        "para_len_median": lens[len(lens) // 2] if lens else 0,
        "para_len_max": lens[-1] if lens else 0,
        "removed_lines": dict(lay.removed),
    }


def run(doc_id: str) -> tuple[list[Unit], dict]:
    lay = extract_layout(str(RAW_DIR / f"{doc_id}.pdf"))
    prof = profile_for(doc_id)
    if prof is not None:
        units = parse(doc_id, lay, prof)                      # MAS：按条款编号
    else:
        from kb_parser_generic import generic_profile_for, parse_generic
        gp = generic_profile_for(doc_id)
        if gp is None:
            raise ValueError(f"{doc_id} 没有对应的解析规则")
        units = parse_generic(doc_id, lay, gp)                # 其他：按排版信号
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{doc_id}.jsonl", "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")
    return units, quality_report(units, lay)


def main():
    ids = sys.argv[1:] or sorted(p.stem for p in RAW_DIR.glob("*.pdf"))
    for doc_id in ids:
        units, rep = run(doc_id)
        print(f"\n=== {doc_id} ===")
        for k, v in rep.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()