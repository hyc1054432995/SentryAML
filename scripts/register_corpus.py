"""
SentryAML - RAG 层 R1：知识库文件登记
=====================================
读取 config/corpus.yaml，检查 data/corpus/raw/ 下的 PDF，把每份文件的
版本、生效区间、内容哈希、页数登记进 Postgres 的 kb_documents 表。

检查项（任何一项不通过，这份文件就不登记）：
  1. 清单字段齐全，没有 TODO（版本信息必须人工核对过）
  2. 文件存在，而且真的是 PDF（文件头是 %PDF-）。
     浏览器下载失败时，常常会把一个 HTML 错误页存成 .pdf，这一步能拦住它。
  3. PDF 能被打开并读出页数
  4. 同一 family 的生效区间不重叠（数据库也有 EXCLUDE 约束兜底）

登记逻辑（幂等，可以反复运行）：
  - 新文件           → 插入，状态 registered
  - 哈希没变         → 只更新清单里的元数据，状态不变
  - 哈希变了         → 更新，状态改为 changed（后续步骤只重建这一份的索引）
  - 数据库有、清单没有 → 只提示，不删除

用法：
    python scripts/register_corpus.py --dry-run     # 只检查，不写数据库
    python scripts/register_corpus.py
"""

import argparse
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

import yaml
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "config" / "corpus.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "corpus" / "raw"

REQUIRED = ["doc_id", "family", "title", "issuer", "doc_type", "jurisdiction",
            "version_label", "effective_from", "source_url"]
DATE_FIELDS = ["issued_date", "effective_from", "effective_to"]
DOC_TYPES = {"notice", "guidelines", "guidance", "typology", "str_guidance", "dataset_paper"}
JURISDICTIONS = {"SG", "INTL"}


def sha256_of(path: Path) -> str:
    """分块读取计算哈希，大文件也不会一次性读进内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_entry(e: dict) -> list[str]:
    errs = []
    for k in REQUIRED:
        if e.get(k) in (None, ""):
            errs.append(f"缺少字段 {k}")
    for k, v in e.items():
        if isinstance(v, str) and v.strip().upper() == "TODO":
            errs.append(f"字段 {k} 还是 TODO，需要核对后填写")
    for k in DATE_FIELDS:
        v = e.get(k)
        if v is not None and not isinstance(v, date) and str(v).upper() != "TODO":
            errs.append(f"{k} 不是合法日期（写成 YYYY-MM-DD）：{v!r}")
    if e.get("doc_type") not in DOC_TYPES:
        errs.append(f"doc_type 只能是 {sorted(DOC_TYPES)}")
    if e.get("jurisdiction") not in JURISDICTIONS:
        errs.append(f"jurisdiction 只能是 {sorted(JURISDICTIONS)}")
    ef, et = e.get("effective_from"), e.get("effective_to")
    if isinstance(ef, date) and isinstance(et, date) and et <= ef:
        errs.append("effective_to 必须晚于 effective_from")
    return errs


def inspect_pdf(path: Path) -> tuple[dict | None, str | None]:
    with open(path, "rb") as f:
        head = f.read(5)
    if head != b"%PDF-":
        return None, f"文件头是 {head!r}，不是 PDF（可能下载到了网页或错误页）"
    try:
        reader = PdfReader(str(path))
        n_pages = len(reader.pages)
        first = reader.pages[0].extract_text() or ""
        last = reader.pages[-1].extract_text() or "" if n_pages > 1 else ""   # 网页存档的日期常在页尾
    except Exception as ex:                                   # 加密、损坏等
        return None, f"PDF 无法读取：{type(ex).__name__}: {ex}"
    flat = re.sub(r"\s+", " ", first).strip()
    snippet = flat[:160]
    # 封面上的版本行单独找出来：只看前 160 个字，会漏掉排在后面的 "Last revised on ..."
    flat_all = flat + " " + re.sub(r"\s+", " ", last)
    version_lines = re.findall(r"(?:last (?:revised|updated) on|revised on|version)\s*[^()]{0,40}?\d{4}",
                               flat_all, re.I)
    return {"n_pages": n_pages, "n_bytes": path.stat().st_size, "sha256": sha256_of(path),
            "snippet": snippet, "first_page": flat_all, "version_lines": version_lines}, None


MONTHS = "january february march april may june july august september october november december".split()


def label_dates_found(label: str, first_page: str) -> list[str]:
    """version_label 里写的日期（如 '30 June 2025'），在首页和末页原文里找不到的列出来。"""
    missing = []
    for d, mon, y in re.findall(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", label):
        if mon.lower() in MONTHS and not re.search(rf"\b{d}\s+{mon}\s+{y}\b", first_page, re.I):
            missing.append(f"{d} {mon} {y}")
    return missing


def overlaps(entries: list[dict]) -> list[str]:
    """同一 family 内，两两检查生效区间是否重叠。"""
    errs = []
    by_fam: dict[str, list[dict]] = {}
    for e in entries:
        by_fam.setdefault(e["family"], []).append(e)
    far = date(9999, 12, 31)
    for fam, es in by_fam.items():
        es = sorted(es, key=lambda x: x["effective_from"])
        for a, b in zip(es, es[1:]):
            if (a.get("effective_to") or far) > b["effective_from"]:
                errs.append(f"{fam}：{a['doc_id']} 与 {b['doc_id']} 的生效区间重叠")
    return errs


def main():
    ap = argparse.ArgumentParser(description="知识库文件登记")
    ap.add_argument("--dry-run", action="store_true", help="只检查，不写数据库")
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        entries = yaml.safe_load(f)["documents"]
    ids = [e["doc_id"] for e in entries]
    assert len(ids) == len(set(ids)), "清单里有重复的 doc_id"

    ok, problems, skipped = [], [], []
    for e in entries:
        path = RAW_DIR / f"{e['doc_id']}.pdf"
        if not path.exists():
            (skipped if e.get("optional") else problems).append(
                (e["doc_id"], [f"找不到文件 {path.relative_to(PROJECT_ROOT)}"]))
            continue
        errs = validate_entry(e)
        info, perr = inspect_pdf(path)
        if perr:
            errs.append(perr)
        if errs:
            problems.append((e["doc_id"], errs))
            continue
        missing_dates = label_dates_found(str(e.get("version_label", "")), info["first_page"])
        if missing_dates:
            problems.append((e["doc_id"], [
                f"version_label 里的日期 {missing_dates} 在首页和末页都找不到；"
                f"文件里的版本行是 {info['version_lines'] or '（没找到）'}，请核对"]))
            continue
        e = {**e, **info, "file_path": path.relative_to(PROJECT_ROOT).as_posix()}
        ok.append(e)

    for msg in overlaps(ok):
        problems.append(("(区间检查)", [msg]))

    # ---------------- 报告 ----------------
    print(f"清单共 {len(entries)} 份：可登记 {len(ok)}，有问题 {len(problems)}，"
          f"可选文件未下载 {len(skipped)}\n")
    for e in ok:
        print(f"  ✓ {e['doc_id']:<22}{e['n_pages']:>4} 页  {e['n_bytes'] / 1024:>8,.0f} KB  "
              f"sha256 {e['sha256'][:12]}…")
        print(f"      版本：{e['version_label']}｜生效 {e['effective_from']} → {e.get('effective_to') or '至今'}")
        print(f"      第一页：{e['snippet']}")
        if e["version_lines"]:
            print(f"      文件里的版本行：{e['version_lines']}")
    for doc_id, errs in problems:
        print(f"  ✗ {doc_id}")
        for m in errs:
            print(f"      - {m}")
    for doc_id, errs in skipped:
        print(f"  · {doc_id}（可选）：{errs[0]}")
    print("\n  请对照上面每份文件的\"第一页\"，确认版本信息和清单里写的一致。")

    if args.dry_run or not ok:
        if problems:
            sys.exit(1)
        return

    # ---------------- 写入数据库 ----------------
    from db import connect
    cols = ["doc_id", "family", "title", "issuer", "doc_type", "jurisdiction", "version_label",
            "issued_date", "effective_from", "effective_to", "source_url", "file_path",
            "sha256", "n_pages", "n_bytes"]
    counts = {"new": 0, "changed": 0, "unchanged": 0}
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, sha256 FROM kb_documents")
        existing = dict(cur.fetchall())
        for e in ok:
            old = existing.get(e["doc_id"])
            kind = "new" if old is None else ("unchanged" if old == e["sha256"] else "changed")
            counts[kind] += 1
            vals = [e.get(c) for c in cols]
            cur.execute(
                f"""INSERT INTO kb_documents ({', '.join(cols)}, status)
                    VALUES ({', '.join(['%s'] * len(cols))}, 'registered')
                    ON CONFLICT (doc_id) DO UPDATE SET
                      {', '.join(f'{c} = EXCLUDED.{c}' for c in cols[1:])},
                      status = CASE WHEN kb_documents.sha256 <> EXCLUDED.sha256
                                    THEN 'changed' ELSE kb_documents.status END,
                      updated_at = now()""",
                vals)
        orphans = sorted(set(existing) - set(ids))
        cur.execute("SELECT status, COUNT(*) FROM kb_documents GROUP BY status ORDER BY status")
        by_status = cur.fetchall()
        conn.commit()

    print(f"\n已写入数据库：新增 {counts['new']}，内容变化 {counts['changed']}，未变 {counts['unchanged']}")
    print("  表中各状态：" + "，".join(f"{s} {n}" for s, n in by_status))
    if orphans:
        print(f"  注意：数据库里有、清单里已没有的文件（未删除）：{', '.join(orphans)}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()