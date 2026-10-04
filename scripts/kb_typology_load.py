"""
SentryAML - RAG 层 R3：手法知识图谱的校验与入库
================================================
读 config/typology_kg.yaml，做五类校验，全部通过才写库（整张图一个事务替换）：

  1. 结构：编号唯一；手法引用的风险信号 / 数据信号、信号引用的工具都存在；没有孤立节点
  2. 出处：每条引用（手法定义 + 风险信号）都要在知识库原文里逐字找到
       - 条款必须存在于该文件族的"当前有效版本"
       - 引文必须是该条款原文的子串（只做排版归一化：弯引号、脚注标记、空白）
       - 给了列表项编号（如 "iv)"）的，引文前面最近的同类编号必须就是它——防止把 iii) 的话标成 iv)
       - 同时记录它在哪些历史版本里也成立（verified_in）
  3. 特征：信号里写的特征名必须在 feature_spec 里存在
  4. 覆盖：IBM 数据集的 8 种模式都要有对应手法；每种手法至少一条来自监管或 FATF 的风险信号、
           至少一个能用工具验证的数据信号
  5. 打印覆盖矩阵，供人审阅

为什么这么严格：这张图是给 Agent 当"监管依据"用的。LLM 抽取或凭记忆整理都会产生
看起来合理、但原文里没有的句子——在合规场景里，引用一句不存在的监管条文比不引用更糟。
所以图由人整理，但每条边都由程序对照原文核对。

用法：
    python scripts/kb_typology_load.py              # 校验并入库
    python scripts/kb_typology_load.py --check      # 只校验，不写库
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import yaml

import feature_spec
from db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KG_PATH = PROJECT_ROOT / "config" / "typology_kg.yaml"

IBM_PATTERNS = {"FAN-IN", "FAN-OUT", "GATHER-SCATTER", "SCATTER-GATHER",
                "CYCLE", "RANDOM", "BIPARTITE", "STACK"}
DATA_SOURCES = {"ibm_aml_paper"}                   # 数据集自身的论文：可作定义，但不算"监管依据"


def norm(s: str) -> str:
    s = re.sub(r"\[\^\d+\]", "", s)
    s = s.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    return re.sub(r"\s+", " ", s).strip().lower()


def marker_regex(item: str) -> re.Pattern | None:
    """列表项编号的"同类编号"正则：iv) → 任意罗马数字加右括号；(b) → 任意括号字母；Step 2 → Step 加数字。"""
    head = r"(?m)^(?:[•\-–]\s*)?"
    if re.fullmatch(r"[ivxl]+\)", item):
        return re.compile(head + r"([ivxl]{1,6}\))")
    if re.fullmatch(r"\([a-z]{1,2}\)", item):
        return re.compile(head + r"(\([a-z]{1,2}\))")
    m = re.fullmatch(r"([A-Za-z]+) \d+", item)
    if m:
        return re.compile(head + rf"({m.group(1)} \d+)")
    return None


def check_citation(cite: dict, units: dict, current: dict, versions: dict) -> tuple[list[str], list[str]]:
    """返回 (错误列表, 通过核对的文件版本列表)。"""
    fam, uid, quote, item = cite["family"], cite["unit_id"], cite["quote"], cite.get("item")
    if fam not in current:
        return [f"文件族 {fam} 没有当前有效版本"], []
    errors, ok_in = [], []
    for doc_id in versions[fam]:
        text = units.get((doc_id, uid))
        if text is None:
            continue
        # 在"原文"上定位，便于检查列表项编号；归一化只影响空白和引号，不改变先后顺序
        t_norm, q_norm = norm(text), norm(quote)
        pos = t_norm.find(q_norm)
        if pos < 0:
            continue
        if item:
            rx = marker_regex(item)
            raw_pos = _raw_position(text, quote)
            if rx is None:
                if item not in text[:raw_pos]:
                    continue
            else:
                marks = [m.group(1) for m in rx.finditer(text[:raw_pos])]
                if not marks or marks[-1] != item:
                    if doc_id == current[fam]:
                        errors.append(f"{fam}:{uid} 引文前最近的编号是 {marks[-1] if marks else '无'}，不是 {item}")
                    continue
        ok_in.append(doc_id)
    if current[fam] not in ok_in and not errors:
        if (current[fam], uid) not in units:
            errors.append(f"{fam}:{uid} 在当前版本 {current[fam]} 中不存在")
        else:
            errors.append(f"{fam}:{uid} 当前版本中找不到引文「{quote[:50]}」")
    return errors, ok_in


def _raw_position(text: str, quote: str) -> int:
    """引文在原文中的起始位置（容忍空白与引号差异）。"""
    pat = r"\s+".join(re.escape(w) for w in norm(quote).split())
    clean = text.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    m = re.search(pat, re.sub(r"\[\^\d+\]", "", clean), re.I)
    return m.start() if m else len(text)


def validate(kg: dict, units: dict, current: dict, versions: dict):
    errors, warnings = [], []
    T = {t["id"]: t for t in kg["typologies"]}
    I = {i["id"]: i for i in kg["indicators"]}
    S = {s["id"]: s for s in kg["signals"]}
    W = {w["id"]: w for w in kg["tools"]}

    # ---- 1. 结构
    for kind, items in (("typology", kg["typologies"]), ("indicator", kg["indicators"]),
                        ("signal", kg["signals"]), ("tool", kg["tools"])):
        dup = [k for k, n in Counter(x["id"] for x in items).items() if n > 1]
        if dup:
            errors.append(f"{kind} 编号重复：{dup}")
    for t in T.values():
        errors += [f"手法 {t['id']} 引用了不存在的风险信号 {x}" for x in t["indicators"] if x not in I]
        errors += [f"手法 {t['id']} 引用了不存在的数据信号 {x}" for x in t["signals"] if x not in S]
    for s in S.values():
        errors += [f"信号 {s['id']} 引用了不存在的工具 {x}" for x in s["tools"] if x not in W]
    used_i = {x for t in T.values() for x in t["indicators"]}
    used_s = {x for t in T.values() for x in t["signals"]}
    used_w = {x for s in S.values() for x in s["tools"]}
    errors += [f"孤立的风险信号：{x}" for x in I if x not in used_i]
    errors += [f"孤立的数据信号：{x}" for x in S if x not in used_s]
    errors += [f"孤立的工具：{x}" for x in W if x not in used_w]

    # ---- 2. 出处
    verified = {}
    cites = [(f"手法 {t['id']} 的定义", t["definition"], ("typ", t["id"])) for t in T.values()] + \
            [(f"风险信号 {i['id']}", i, ("ind", i["id"])) for i in I.values()]
    for label, cite, key in cites:
        errs, ok_in = check_citation(cite, units, current, versions)
        errors += [f"{label}：{e}" for e in errs]
        verified[key] = ok_in

    # ---- 3. 特征
    known = set(feature_spec.NUM_FEATURES) | set(feature_spec.CAT_FEATURES)
    for s in S.values():
        errors += [f"信号 {s['id']} 的特征 {f} 不在 feature_spec 里" for f in s["features"] if f not in known]
        if not s["features"] and not s["tools"]:
            errors.append(f"信号 {s['id']} 既没有特征也没有工具，无法观测")

    # ---- 4. 覆盖
    covered = {t["dataset_pattern"] for t in T.values() if t.get("dataset_pattern")}
    if IBM_PATTERNS - covered:
        errors.append(f"IBM 数据集模式未覆盖：{sorted(IBM_PATTERNS - covered)}")
    for t in T.values():
        if not any(I[x]["family"] not in DATA_SOURCES for x in t["indicators"] if x in I):
            errors.append(f"手法 {t['id']} 没有来自监管或 FATF 的风险信号")
        if not any(S[x]["tools"] for x in t["signals"] if x in S):
            errors.append(f"手法 {t['id']} 没有可用工具验证的数据信号")
    for key, ok_in in verified.items():
        if ok_in and len(ok_in) < len(versions[_family_of(key, T, I)]):
            warnings.append(f"{key[1]}：只在 {ok_in} 中成立（其他版本的该条款措辞不同或不存在）")
    return errors, warnings, verified


def _family_of(key, T, I):
    return T[key[1]]["definition"]["family"] if key[0] == "typ" else I[key[1]]["family"]


def print_matrix(kg: dict) -> None:
    I = {i["id"]: i for i in kg["indicators"]}
    S = {s["id"]: s for s in kg["signals"]}
    fams = sorted({i["family"] for i in I.values()})
    short = {f: f.replace("mas_n626_guidelines", "MAS指引").replace("mas_tm_guidance", "MAS-TM")
             .replace("fatf_", "FATF-").replace("ibm_aml_paper", "IBM") for f in fams}
    print("\n覆盖矩阵（风险信号条数 / 数据信号）")
    print(f"  {'手法':<28}" + "".join(f"{short[f]:>10}" for f in fams) + "   数据信号（* = 有特征列，否则只能靠工具）")
    for t in kg["typologies"]:
        cnt = Counter(I[x]["family"] for x in t["indicators"])
        sig = ", ".join(x.removeprefix("s_") + ("*" if S[x]["features"] else "") for x in t["signals"])
        print(f"  {t['id']:<28}" + "".join(f"{cnt.get(f, 0) or '·':>10}" for f in fams) + f"   {sig}")
    by_sig = Counter(x for t in kg["typologies"] for x in t["signals"])
    print("\n  被多种手法共用的信号（不能用来判断是哪种手法，排序时降权）："
          + "，".join(f"{k.removeprefix('s_')}×{n}" for k, n in by_sig.most_common() if n >= 3))
    tool_only = [s["id"] for s in kg["signals"] if not s["features"]]
    print("  模型特征里没有、只能靠 Agent 工具观测的信号：" + "，".join(x.removeprefix("s_") for x in tool_only))


def main():
    ap = argparse.ArgumentParser(description="手法知识图谱：校验并入库")
    ap.add_argument("--check", action="store_true", help="只校验，不写库")
    args = ap.parse_args()

    with open(KG_PATH, encoding="utf-8") as f:
        kg = yaml.safe_load(f)

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, family FROM kb_documents_current")
        current = {fam: d for d, fam in cur.fetchall()}
        cur.execute("SELECT family, array_agg(doc_id ORDER BY effective_from DESC) FROM kb_documents GROUP BY family")
        versions = dict(cur.fetchall())
        cur.execute("SELECT doc_id, unit_id, text FROM kb_units")
        units = {(d, u): t for d, u, t in cur.fetchall()}

    errors, warnings, verified = validate(kg, units, current, versions)
    n_cite = len(kg["typologies"]) + len(kg["indicators"])
    print(f"手法 {len(kg['typologies'])}，风险信号 {len(kg['indicators'])}，数据信号 {len(kg['signals'])}，"
          f"工具 {len(kg['tools'])}；引用原文 {n_cite} 处")
    print_matrix(kg)
    if warnings:
        print("\n提示（不阻止入库）：")
        for w in warnings:
            print(f"  · {w}")
    if errors:
        print(f"\n校验未通过（{len(errors)} 项），不写数据库：")
        for e in errors:
            print(f"  ✗ {e}")
        raise SystemExit(1)
    print(f"\n校验通过：{n_cite} 处引文全部在当前有效版本的原文中逐字找到，列表项编号一致")
    if args.check:
        return

    with connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE kb_typology_indicators, kb_typology_signals, kb_signal_tools, "
                    "kb_typologies, kb_indicators, kb_signals, kb_tools")
        cur.executemany(
            """INSERT INTO kb_typologies (typology_id, name, dataset_pattern, stage, description,
                                          def_family, def_unit_id, def_item, def_quote, def_verified_in)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [(t["id"], t["name"], t.get("dataset_pattern"), t["stage"], t["description"].strip(),
              t["definition"]["family"], t["definition"]["unit_id"], t["definition"].get("item"),
              t["definition"]["quote"], verified[("typ", t["id"])]) for t in kg["typologies"]])
        cur.executemany(
            "INSERT INTO kb_indicators VALUES (%s, %s, %s, %s, %s, %s)",
            [(i["id"], i["family"], i["unit_id"], i.get("item"), i["quote"], verified[("ind", i["id"])])
             for i in kg["indicators"]])
        cur.executemany("INSERT INTO kb_signals VALUES (%s, %s, %s, %s, %s)",
                        [(s["id"], s["name"], s["features"], s["evidence"], s["caution"]) for s in kg["signals"]])
        cur.executemany("INSERT INTO kb_tools VALUES (%s, %s, %s, %s)",
                        [(w["id"], w["status"], w.get("backing"), w["description"]) for w in kg["tools"]])
        cur.executemany("INSERT INTO kb_typology_indicators VALUES (%s, %s)",
                        [(t["id"], x) for t in kg["typologies"] for x in t["indicators"]])
        cur.executemany("INSERT INTO kb_typology_signals VALUES (%s, %s)",
                        [(t["id"], x) for t in kg["typologies"] for x in t["signals"]])
        cur.executemany("INSERT INTO kb_signal_tools VALUES (%s, %s)",
                        [(s["id"], x) for s in kg["signals"] for x in s["tools"]])
        conn.commit()
    print("已写入 kb_typologies / kb_indicators / kb_signals / kb_tools 及三张边表")


if __name__ == "__main__":
    main()