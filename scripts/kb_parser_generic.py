"""
SentryAML - RAG 层 R2c：通用结构解析（FATF 报告、STR 指南、网页、论文）
======================================================================
MAS 文件每段都有编号，靠编号就能切（kb_parser.py）。这里的文件没有统一编号，
所以换一套思路：靠"排版信号"判断段落和标题。

  标题     加粗、或字号明显大于正文，且较短、不以句号结尾
  新段落   与上一行的垂直间距明显大于正常行距（行距的 1.35 倍）
           或者行首是段落编号（FATF CEF 报告有 "25." 这样的全局编号）
  列表     以 ":" 结尾的引导句后面的项目符号，归入同一段（引导句 + 列表是一个完整意思）；
           直接挂在标题下的项目符号，每条单独成段（例如 Annex A 的风险信号，一条一个）
  方框     "Box 9. ..." 开始，到 "Source: ..." 或缩进恢复时结束，整体成一个单元（案例研究）
  跨页     上一页最后一行没有以句末标点结束 → 下一页第一行是续行

单元编号：
  有段落编号的用编号（"25"）；方框用 "box-9"；其余用 "页码.序号"（"p18.3" = 第 18 页第 3 段），
  引用时可以写成 "FATF (2018) p.18"，这是没有条款编号的报告的标准引用方式。

用法：由 kb_parser.py 自动调用；也可以单独运行：
    python scripts/kb_parser_generic.py fatf_cef_2023
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

from kb_layout import DocLayout, Line
from kb_parser import ENUM_RE, Unit, _join, _unique

GENERIC_PROFILES = {
    "fatf": dict(
        para_num_re=r"^(\d{1,3})\.\s+(?=\S)",          # CEF 报告的全局段落编号 "25. ..."
        heading_num_re=r"^((?:\d+\.)*\d+)\.?\s+(\S.*)$",
        box_re=r"^Box (\d+)[.:]\s*(.*)$",
        drop_small=2.5, keep_small_re=r"^Source",
        drop_line_re=None, stop_re=r"^(REFERENCES|References|Bibliography)$",   # 参考文献之后不进知识库
        page_units=False,
    ),
    "paper": dict(
        para_num_re=None,
        heading_num_re=r"^((?:\d+\.)*\d+|[A-G](?:\.\d+)?)\s+(\S.*)$",
        box_re=None,
        drop_small=1.5, keep_small_re=None,              # 图里的小字（坐标轴、节点标签）不要
        drop_line_re=r"^arXiv:\d",                       # 页边竖排的 arXiv 编号
        stop_re=r"^Acknowledg",                          # 致谢之后是参考文献和附录，不进知识库
        page_units=False,
    ),
    "web": dict(
        para_num_re=None, heading_num_re=None, box_re=None,
        drop_small=0.5, keep_small_re=None,              # 导航栏、面包屑都是小字
        drop_line_re=r"^(BACK TO TOP|Share This Content|Home /|A Singapore Government Agency)",
        stop_re=None, page_units=False,
    ),
    "slides": dict(page_units=True, drop_small=3.0, keep_small_re=None, drop_line_re=None),
}
GENERIC_OF = {
    "fatf_": "fatf",
    "ibm_aml_paper": "paper",
    "spf_stro_page": "web",
    "spf_str_form_guide": "slides",
}

SENT_END = tuple('.!?:;"”)')


def generic_profile_for(doc_id: str) -> str | None:
    for prefix, prof in GENERIC_OF.items():
        if doc_id.startswith(prefix):
            return prof
    return None


def _line_spacing(lines: list[Line], body: float) -> float:
    """正文的典型行距：同一页、正文字号、相邻两行基线之差的中位数。"""
    d = []
    for a, b in zip(lines, lines[1:]):
        if a.page == b.page and abs(a.size - body) < 0.6 and abs(b.size - body) < 0.6:
            dy = b.y0 - a.y0
            if 0.5 * body < dy < 2.5 * body:
                d.append(dy)
    return statistics.median(d) if d else body * 1.2


def _keep(ln: Line, P: dict, body: float) -> bool:
    t = ln.text.strip()
    if not re.search(r"[A-Za-z0-9]", t):
        return False                                   # 图标、装饰符号
    if P.get("drop_line_re") and re.search(P["drop_line_re"], t):
        return False
    if ln.size < body - P["drop_small"] and not (P.get("keep_small_re") and re.match(P["keep_small_re"], t)):
        return False
    return True


# ---------------------------------------------------------------------------
# 按页成块（STR 填表指南：每页是一张说明"幻灯片"）
# ---------------------------------------------------------------------------
def _parse_pages(doc_id: str, lines: list[Line], body: float) -> list[Unit]:
    units = []
    by_page: dict[int, list[Line]] = {}
    for ln in lines:
        by_page.setdefault(ln.page, []).append(ln)
    for page, ls in sorted(by_page.items()):
        title_ln = max(ls, key=lambda l: l.size)
        title = title_ln.text if title_ln.size >= body + 4 else f"Page {page}"
        rest = [l for l in ls if l is not title_ln]
        text = ""
        for l in sorted(rest, key=lambda l: (l.y0, l.x0)):
            text = _join(text, l.text.strip())
        if text:
            units.append(Unit(doc_id, f"p{page}", "block", [title], text, page, page))
    return units


# ---------------------------------------------------------------------------
# 通用解析
# ---------------------------------------------------------------------------
def parse_generic(doc_id: str, lay: DocLayout, prof_name: str) -> list[Unit]:
    P = GENERIC_PROFILES[prof_name]
    body = lay.body_size
    lines = [ln for ln in lay.lines if _keep(ln, P, body)]
    if P["page_units"]:
        units = _parse_pages(doc_id, lines, body)
        _attach_footnotes(doc_id, units, lay)
        return units

    spacing = _line_spacing(lines, body)
    gap_thr = spacing * 1.35
    para_re = re.compile(P["para_num_re"]) if P["para_num_re"] else None
    head_re = re.compile(P["heading_num_re"]) if P["heading_num_re"] else None
    box_re = re.compile(P["box_re"]) if P["box_re"] else None
    stop_re = re.compile(P["stop_re"]) if P["stop_re"] else None

    units: list[Unit] = []
    heads: list[tuple[int, str, str]] = []
    cur: Unit | None = None
    prev: Line | None = None
    page_k: Counter = Counter()
    last_num = 0
    in_box = False
    box_x: float | None = None
    list_mode = False
    last_heading: Line | None = None

    def path():
        return [f"{hid} {t}".strip() for _, hid, t in heads]

    def new_unit(uid, kind, text, ln, parent=None):
        nonlocal cur, list_mode
        cur = Unit(doc_id, _unique(uid, units), kind, path(), text, ln.page, ln.page,
                   parent, list(ln.footnote_refs))
        units.append(cur)
        list_mode = False
        return cur

    def auto_id(ln):
        page_k[ln.page] += 1
        return f"p{ln.page}.{page_k[ln.page]}"

    def push_heading(level, hid, title, ln):
        nonlocal cur
        while heads and heads[-1][0] >= level:
            heads.pop()
        heads.append((level, hid, title))
        units.append(Unit(doc_id, _unique(f"§{hid or len(units)}", units), "heading", path()[:-1],
                          f"{hid} {title}".strip(), ln.page, ln.page, None, list(ln.footnote_refs)))
        cur = None

    for ln in lines:
        t = ln.text.strip()
        gap = (ln.y0 - prev.y0) if (prev and prev.page == ln.page) else None
        looks_heading = ((ln.bold or ln.size >= body + 1.5) and len(t) < 160
                         and not t.endswith((".", ";", ",")) and not t.startswith("•"))

        if stop_re and looks_heading and stop_re.match(t):
            break

        # ---- 方框开始
        if box_re and (m := box_re.match(t)) and (ln.bold or ln.size > body):
            new_unit(f"box-{m.group(1)}", "box", t, ln, parent=heads[-1][1] if heads else None)
            in_box, box_x, prev, last_heading = True, None, ln, None
            continue

        # ---- 方框内部
        if in_box:
            if box_x is None and ln.bold and looks_heading:          # 方框标题折行
                cur.text = f"{cur.text} {t}"
                prev = ln
                continue
            ends_by_indent = box_x is not None and ln.x0 < box_x - 6 and not t.startswith("•")
            if not (looks_heading or ends_by_indent or (para_re and para_re.match(t))):
                if box_x is None:
                    box_x = ln.x0
                cur.text = _join(cur.text, t) if not (gap and gap > gap_thr) else cur.text + "\n" + t
                cur.page_end = ln.page
                cur.footnote_refs.extend(ln.footnote_refs)
                prev = ln
                if t.startswith("Source"):
                    in_box, cur = False, None                  # 案例框以 "Source: 某国" 结尾
                continue
            in_box, cur = False, None

        # ---- 标题
        if looks_heading:
            # 紧接上一行标题、间距小 → 标题折行，合并
            if last_heading is not None and ln.page == last_heading.page and gap is not None \
                    and gap < max(spacing * 1.6, ln.size * 1.5) and heads \
                    and abs(ln.size - last_heading.size) < 0.6:
                lv, hid, title = heads[-1]
                heads[-1] = (lv, hid, f"{title} {t}")
                units[-1].text = f"{units[-1].text} {t}"
                last_heading, prev = ln, ln
                continue
            m = head_re.match(t) if head_re else None
            if m:
                hid, title = m.group(1), m.group(2)
                level = hid.count(".") + 1
            else:
                hid, title = "", t
                deepest = max((lv for lv, h, _ in heads if h), default=0)
                level = 1 if ln.size >= body + 4 else deepest + 1
            push_heading(level, hid, title, ln)
            last_heading, prev = ln, ln
            continue
        last_heading = None

        # ---- 带编号的段落（全局递增）
        if para_re and (m := para_re.match(t)):
            n = int(m.group(1))
            if n > last_num:
                last_num = n
                new_unit(str(n), "paragraph", t[m.end():].strip(), ln)
                prev = ln
                continue

        # ---- 新段落还是续行
        if cur is None:
            start_new = True
        elif gap is None:                                      # 换页
            start_new = cur.text.rstrip().endswith(SENT_END) and not t[:1].islower()
        else:
            start_new = gap > gap_thr
        if start_new and t.startswith("•") and cur is not None and \
                (cur.text.rstrip().endswith(":") or list_mode):
            start_new = False                                  # 引导句后面的列表，归入同一段
            list_mode = True
        if start_new:
            new_unit(auto_id(ln), "paragraph", t, ln)
        else:
            cur.text = _join(cur.text, t) if not t.startswith("•") else cur.text + "\n" + t
            cur.page_end = ln.page
            cur.footnote_refs.extend(ln.footnote_refs)
        prev = ln

    _attach_footnotes(doc_id, units, lay)
    for i, u in enumerate(units):
        u.order = i
        u.text = u.text.strip()
        u.footnote_refs = sorted(set(u.footnote_refs))
    return units


def _attach_footnotes(doc_id: str, units: list[Unit], lay: DocLayout) -> None:
    owner = {}
    for u in units:
        for n in u.footnote_refs:
            owner.setdefault(n, u)
    for fn in lay.footnotes:
        o = owner.get(fn.num)
        units.append(Unit(doc_id, _unique(f"fn{fn.num}", units), "footnote",
                          (o.heading_path + [o.unit_id]) if o else [], fn.text, fn.page, fn.page,
                          o.unit_id if o else None))


if __name__ == "__main__":
    import sys
    from kb_parser import run
    for d in sys.argv[1:]:
        units, rep = run(d)
        print(d, rep)