"""
SentryAML - RAG 层 R2a：PDF 版面提取与清洗
==========================================
把 PDF 变成"带版面特征的文本行"，并去掉不属于正文的东西。
这一层不理解任何条款结构，只负责给后面的结构解析器（kb_parser.py）提供干净的输入。

为什么不直接用 page.get_text() 拿纯文本：
  纯文本丢掉了判断结构所需的全部线索——字号、粗体、左边距、上下间距、上标。
  例如 MAS Notice 里 "14.1" 和它的正文在 PDF 里是两个独立的文本块，
  只有按纵坐标把它们拼回同一行，才能知道 "14.1" 是这段话的编号。

做五件事：
  1. 视觉行重建：把纵坐标重叠的文本块拼成一行，保留字号、粗体、x 坐标
  2. 上标识别：正文里比周围字号小得多的数字是脚注引用，改写成 [^14]
  3. 页眉页脚去除：在页面顶部/底部区域、且在多数页面上重复出现的文字（数字归一后比较）
  4. 目录去除：带点状引导线（.....）的行
  5. 脚注分离：页面下部、以小号数字开头、字号小于正文的行，归入脚注，不进入正文

依赖：pip install pymupdf
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

BULLET_CHARS = "••▪●–‣"   # 各种字体里的项目符号
TOC_RE = re.compile(r"(\.\s?){6,}\s*\d*\s*$|…{3,}|\.{2,}\s*\d{1,3}\s*$")   # 点状引导线 + 页码
FOOTNOTE_START_RE = re.compile(r"^\[\^(\d+)\]\s*(.*)$")


@dataclass
class Line:
    page: int                    # 从 1 开始
    x0: float
    y0: float
    x1: float
    y1: float
    size: float                  # 这一行的主字号（按字符数加权，不含上标）
    bold: bool                   # 非空白字符全部为粗体
    text: str                    # 已把上标数字改写成 [^n]
    footnote_refs: list[int] = field(default_factory=list)


@dataclass
class Footnote:
    num: int
    page: int
    text: str


@dataclass
class DocLayout:
    doc_path: str
    n_pages: int
    body_size: float
    lines: list[Line]            # 正文行，按页、按纵坐标排序
    footnotes: list[Footnote]
    removed: Counter             # 各类被去除的行数，用于质量报告


# ---------------------------------------------------------------------------
# 1–2. 视觉行重建 + 上标识别
# ---------------------------------------------------------------------------
def _span_is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "Bold" in span["font"] or "Black" in span["font"]


def _raw_lines(page: pymupdf.Page, pno: int) -> list[dict]:
    """PyMuPDF 的 line（同一文本块内的一行）→ 字典，保留 span 级信息。"""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            x0, _, x1, _ = ln["bbox"]
            # 纵坐标用基线 + 字号重新计算：有些 PDF（如 FATF 报告）的行框高度被字体度量撑大到 50pt，
            # 直接用 bbox 会让行距、页眉区域判断全部失真
            size = max(s["size"] for s in spans)
            base = max(s["origin"][1] for s in spans if s["size"] >= size * 0.8)
            y0, y1 = base - size * 0.8, base + size * 0.2
            out.append({"page": pno, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "yc": base - size * 0.3, "spans": spans})
    return out


def _merge_visual_lines(raw: list[dict], tol: float = 3.0) -> list[list[dict]]:
    """纵向中心相差不超过 tol 的片段属于同一视觉行。"""
    raw = sorted(raw, key=lambda r: (r["yc"], r["x0"]))
    groups: list[list[dict]] = []
    for r in raw:
        if groups and abs(groups[-1][0]["yc"] - r["yc"]) <= tol:
            groups[-1].append(r)
        else:
            groups.append([r])
    # 同一高度、但横向相隔很远且字号差别很大的片段不是同一行（例如左侧大标题与右栏正文）
    out = []
    for g in groups:
        g = sorted(g, key=lambda r: r["x0"])
        seg = [g[0]]
        for r in g[1:]:
            a = max(s["size"] for s in seg[-1]["spans"])
            b = max(s["size"] for s in r["spans"])
            if r["x0"] - seg[-1]["x1"] > 40 and abs(a - b) / max(a, b) > 0.3:
                out.append(seg)
                seg = [r]
            else:
                seg.append(r)
        out.append(seg)
    return out


def _build_line(group: list[dict]) -> Line:
    spans = [s for r in group for s in r["spans"]]
    # 主字号：字符数最多的字号
    size_chars = Counter()
    for s in spans:
        size_chars[round(s["size"], 1)] += len(s["text"].strip())
    main = size_chars.most_common(1)[0][0]

    parts, refs = [], []
    prev_x1 = None
    for s in spans:
        t = s["text"]
        small = s["size"] < main * 0.8
        is_sup = (s["flags"] & 1) or small
        if is_sup and re.fullmatch(r"\s*\d{1,3}\s*[,.;]?\s*", t):
            n = int(re.sub(r"\D", "", t))
            refs.append(n)
            parts.append(f"[^{n}]" + re.sub(r"[\d\s]", "", t))
        else:
            # 不同文本块之间补一个空格（例如编号 "14.1" 与正文）
            if prev_x1 is not None and s["bbox"][0] - prev_x1 > 1.5 and parts and not parts[-1].endswith(" "):
                parts.append(" ")
            parts.append(t)
        prev_x1 = s["bbox"][2]
    text = re.sub(r"\s+", " ", "".join(parts)).strip()

    body_spans = [s for s in spans if s["size"] >= main * 0.8 and s["text"].strip()]
    bold = bool(body_spans) and all(_span_is_bold(s) for s in body_spans)
    return Line(page=group[0]["page"],
                x0=min(r["x0"] for r in group), y0=min(r["y0"] for r in group),
                x1=max(r["x1"] for r in group), y1=max(r["y1"] for r in group),
                size=main, bold=bold, text=text, footnote_refs=refs)


# ---------------------------------------------------------------------------
# 3. 页眉页脚
# ---------------------------------------------------------------------------
def _norm_boiler(t: str) -> str:
    return re.sub(r"\d+", "#", t).strip().lower()


def _find_boilerplate(pages: list[list[Line]], heights: list[float],
                      top: float = 0.10, bottom: float = 0.12) -> set[str]:
    """在多数页面的顶部/底部区域重复出现的文字（数字归一后）。"""
    counts = Counter()
    for lines, h in zip(pages, heights):
        seen = set()
        for ln in lines:
            if ln.y0 < h * top or ln.y1 > h * (1 - bottom):
                seen.add(_norm_boiler(ln.text))
        counts.update(seen)
    n = len(pages)
    return {t for t, c in counts.items() if c >= max(3, 0.3 * n)}


# ---------------------------------------------------------------------------
# 表格：多列表格按行转写成一行文字；单列的"表格"其实是方框（案例框），保持原样
# ---------------------------------------------------------------------------
def _extract_tables(page: pymupdf.Page, pno: int, raw: list[dict]) -> tuple[list[dict], list[Line]]:
    try:
        tabs = page.find_tables().tables
    except Exception:
        return raw, []
    out_lines, keep = [], raw
    for tb in tabs:
        rows = [[re.sub(r"\s+", " ", c or "").strip() for c in r] for r in tb.extract()]
        rows = [[c for c in r if c] for r in rows]
        rows = [r for r in rows if r]
        if sum(1 for r in rows if len(r) >= 2) < 2:
            continue                                  # 单列（方框）或只有一行多列（误检），不当表格处理
        x0, y0, x1, y1 = tb.bbox
        inside = lambda r: (r["x0"] >= x0 - 2 and r["x1"] <= x1 + 2
                            and r["y0"] >= y0 - 2 and r["y1"] <= y1 + 2)
        sizes = sorted(s["size"] for r in keep if inside(r) for s in r["spans"])
        size = round(sizes[len(sizes) // 2], 1) if sizes else 12.0
        keep = [r for r in keep if not inside(r)]
        for k, r in enumerate(rows):
            out_lines.append(Line(page=pno, x0=x0, y0=y0 + k * 0.01, x1=x1, y1=y0 + k * 0.01,
                                  size=size, bold=False, text="• " + " — ".join(r)))
    return keep, out_lines


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def extract_layout(pdf_path: str, skip_pages: set[int] | None = None) -> DocLayout:
    doc = pymupdf.open(pdf_path)
    skip_pages = skip_pages or set()
    pages, heights = [], []
    n_tables = 0
    for i, page in enumerate(doc):
        pno = i + 1
        raw = _raw_lines(page, pno) if pno not in skip_pages else []
        raw, table_lines = _extract_tables(page, pno, raw)
        n_tables += len(table_lines)
        lines = [_build_line(g) for g in _merge_visual_lines(raw)] + table_lines
        lines.sort(key=lambda ln: (ln.y0, ln.x0))
        # 目录页：点状引导线的行占多数 → 整页丢弃
        if sum(1 for ln in lines if TOC_RE.search(ln.text)) >= 5:
            lines = []
        pages.append(lines)
        heights.append(page.rect.height)

    # 正文字号：全文按字符数加权的众数
    sz = Counter()
    for lines in pages:
        for ln in lines:
            sz[ln.size] += len(ln.text)
    body_size = sz.most_common(1)[0][0]

    boiler = _find_boilerplate(pages, heights)
    removed = Counter()
    if n_tables:
        removed["表格行（已转写）"] = n_tables
    body: list[Line] = []
    footnotes: list[Footnote] = []

    for lines, h in zip(pages, heights):
        # ---- 5. 脚注区：页面下部第一条"小号数字开头、字号明显小于正文"的行，及其以下的小字号行
        fn_start_y = None
        for ln in lines:
            if (ln.y0 > h * 0.55 and ln.size <= body_size - 0.5
                    and FOOTNOTE_START_RE.match(ln.text)
                    and _norm_boiler(ln.text) not in boiler):
                fn_start_y = ln.y0
                break
        current_fn = None
        for ln in lines:
            nb = _norm_boiler(ln.text)
            in_margin = ln.y0 < h * 0.07 or ln.y1 > h * 0.93
            if nb in boiler or (in_margin and re.fullmatch(r"#|page #( of #)?", nb)):
                removed["页眉页脚"] += 1
                continue
            if TOC_RE.search(ln.text) or ln.text.strip().lower() in {"table of contents", "contents"}:
                removed["目录"] += 1
                continue
            if fn_start_y is not None and ln.y0 >= fn_start_y - 0.5 and ln.size <= body_size - 0.5:
                m = FOOTNOTE_START_RE.match(ln.text)
                if m:
                    current_fn = Footnote(num=int(m.group(1)), page=ln.page, text=m.group(2))
                    footnotes.append(current_fn)
                elif current_fn is not None:
                    current_fn.text = (current_fn.text + " " + ln.text).strip()
                removed["脚注行"] += 1
                continue
            stripped = ln.text.strip()
            if stripped and all(c in BULLET_CHARS or c.isspace() for c in stripped):
                removed["孤立项目符号"] += 1          # 与文字错位的项目符号，段落分隔靠行距判断
                continue
            if stripped and stripped[0] in BULLET_CHARS:
                ln.text = "• " + stripped[1:].strip()
            body.append(ln)

    return DocLayout(doc_path=pdf_path, n_pages=len(doc), body_size=body_size,
                     lines=body, footnotes=footnotes, removed=removed)


if __name__ == "__main__":                      # 调试：python kb_layout.py 文件.pdf 页码
    import sys
    lay = extract_layout(sys.argv[1])
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(f"正文字号 {lay.body_size}，去除：{dict(lay.removed)}，脚注 {len(lay.footnotes)} 条")
    for ln in lay.lines:
        if ln.page == want:
            print(f"{ln.x0:6.1f} {ln.y0:6.1f} {ln.size:4.1f} {'B' if ln.bold else ' '} {ln.text[:110]}")
    for f in lay.footnotes:
        if f.page == want:
            print(f"  [脚注 {f.num}] {f.text[:100]}")