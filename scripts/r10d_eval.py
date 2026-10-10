"""
SentryAML - R10d：STR 草稿的成批评测（D81）
==========================================
在 R9c 的 dev / test 样本上，对每一条"Agent 升级的 T2 告警"模拟复核员走一遍：
    确认可疑（采用 Agent 给的手法）→ 组装草稿（R10b-1）→ 模型起草理由（R10b-2）
记录每一步的结果，最后**整体回滚**：复核表只追加，评测数据一旦提交就会永远混在真实的复核记录里。
模型调用的缓存在另一个连接里提交、会保留下来，所以重跑结果完全相同、不再花钱。

为什么只评 T2 的 Agent 升级：方案 B 下 T1 不经过 Agent，没有调查证据，草稿只能从告警本身写起，另作说明（D81）。

自动指标：草稿能否组装（数据库检查是否通过）、理由第一次 / 最终是否通过核验、退回原因的分布、长度、费用、耗时。
人工指标：程序检查证明不了"每句话都被所引事实支撑"（D80），所以按固定种子抽 5 份理由，逐句打分：
    python scripts/r10d_eval.py --split dev             # 运行，写出 reports/r10d_dev.json 和评分表 r10d_rubric_dev.yaml
    （人工填写评分表）
    python scripts/r10d_eval.py --split dev --score     # 汇总人工评分
test 只跑一次：结果文件已存在就拒绝；而且 test 必须用 dev 上评测过的同一个提示词版本。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

import psycopg
import yaml

import str_narrative as narr
from db import connect
from review import act, snapshot
from str_draft import BUILDER, build_body, insert_draft, latest, load_case

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT_ROOT / "reports" / "r9c_split.json"
RUNS = {"dev": PROJECT_ROOT / "reports" / "r9b_dev_runs.json", "test": PROJECT_ROOT / "reports" / "r9c_test_runs.json"}
SEED = 20261010
N_RUBRIC = 5
ANALYST = "alice"


def out_path(split: str) -> Path:
    return PROJECT_ROOT / "reports" / f"r10d_{split}.json"


def rubric_path(split: str) -> Path:
    return PROJECT_ROOT / "reports" / f"r10d_rubric_{split}.yaml"


def sentences(text: str) -> list[str]:
    """按句号、问号、感叹号后的空白切句。金额里的小数点后面没有空格，不会被切开。"""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ---------------------------------------------------------------------------
# 选样本
# ---------------------------------------------------------------------------
def select(cur, split: str) -> tuple[list[dict], list[dict]]:
    """返回 (要评测的案件, 排除的告警及原因)。"""
    sp = json.loads(SPLIT.read_text(encoding="utf-8"))
    runs = json.loads(RUNS[split].read_text(encoding="utf-8"))
    ids = [x["alert_id"] for x in sp[split]]
    cur.execute("SELECT alert_id, tier FROM alerts WHERE alert_id = ANY(%s)", (ids,))
    tier = dict(cur.fetchall())
    cur.execute("SELECT typology_id FROM kb_typologies")
    kg = {r[0] for r in cur.fetchall()}
    cur.execute("""SELECT c.alert_id, c.case_id, s.state FROM review_cases c
                   JOIN review_case_state s ON s.case_id = c.case_id WHERE c.alert_id = ANY(%s)""", (ids,))
    existing = {a: (cid, st) for a, cid, st in cur.fetchall()}
    picked, excluded = [], []
    for aid in sorted(ids):
        rec = runs.get(str(aid))
        if tier[aid] != "T2" or rec is None:
            continue
        if rec.get("status") == "done" and rec.get("decision") != "escalate":
            continue                                        # Agent 关闭的：不进 STR
        if rec.get("status") != "done":
            excluded.append({"alert_id": aid, "why": "agent error: no evidence to draft from"})
            continue
        if rec.get("typology") not in kg:
            excluded.append({"alert_id": aid, "why": f"agent typology '{rec.get('typology')}' is not in the "
                                                     "knowledge graph; a reviewer would have to choose one"})
            continue
        if aid in existing and existing[aid][1] != "pending_review":
            excluded.append({"alert_id": aid, "why": f"case {existing[aid][0]} was already processed by hand "
                                                     f"({existing[aid][1]})"})
            continue
        picked.append({"alert_id": aid, "rec": rec, "case_id": existing.get(aid, (None,))[0]})
    return picked, excluded


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
def run(split: str) -> dict:
    t_all = time.perf_counter()
    rows = []
    with connect() as conn, conn.cursor() as cur:
        picked, excluded = select(cur, split)
        cur.execute("SELECT run_id::text FROM agent_runs WHERE run_id::text = ANY(%s)",
                    ([p["rec"].get("run_id") for p in picked],))
        known = {r[0] for r in cur.fetchall()}
        # 还没有案件的告警（test 全部是）：按 review.py open 的方式建案件——同一个快照函数，事务结束时回滚
        for p in picked:
            if p["case_id"] is None:
                rec = p["rec"]
                cur.execute("""INSERT INTO review_cases (alert_id, route, agent_run_id, agent_decision, agent_result)
                               VALUES (%s, 'agent_escalate', %s, 'escalate', %s) RETURNING case_id""",
                            (p["alert_id"], rec["run_id"] if rec.get("run_id") in known else None,
                             json.dumps(snapshot(rec), ensure_ascii=False, default=str)))
                p["case_id"] = cur.fetchone()[0]
                act(cur, p["case_id"], "open", "system")
        cur.execute("SET LOCAL ROLE sentry_review")
        print(f"{split}：评测 {len(picked)} 个案件，排除 {len(excluded)} 个")
        for e in excluded:
            print(f"  排除告警 #{e['alert_id']}：{e['why']}")
        for p in picked:
            cid, rec = p["case_id"], p["rec"]
            row = {"alert_id": p["alert_id"], "case_id": cid, "typology": rec["typology"]}
            t0 = time.perf_counter()
            cur.execute("SAVEPOINT c")
            try:
                act(cur, cid, "confirm_suspicious", ANALYST, payload={"typology": rec["typology"]})
                case = load_case(cur, cid)
                body, csv_text = build_body(cur, case)
                insert_draft(cur, case, ANALYST, body, csv_text, {"kind": "builder", "builder": BUILDER})
                row.update(draft="ok", accounts=len(body["accounts"]), txns=len(body["suspicious_transactions"]),
                           attachment_rows=body["attachment"]["rows"], categories=len(body["reasons"]["categories"]))
            except psycopg.Error as e:
                cur.execute("ROLLBACK TO SAVEPOINT c")
                row.update(draft="rejected", error=(e.diag.message_primary or str(e))[:300])
                rows.append(row)
                print(f"  #{p['alert_id']:<5} 草稿被拒绝：{row['error'][:100]}")
                continue
            r = narr.write(cid, ANALYST, cur=cur)
            if r["status"] == "verified":
                # 每句话里的数字出自哪一步哪个字段：写进评分表，评分的人不必自己去翻工具结果（D82）
                results, _ = narr.materials(cur, load_case(cur, cid), latest(cur, cid))
                cited = {(f["step"], f["path"]) for f in r["facts"]}
                row["sentence_sources"] = [narr.number_sources(sx, results, cited=cited)
                                           for sx in sentences(r["narrative"])]
            row.update(narrative=r["status"], attempts=len(r["attempts"]),
                       first_pass=bool(r["attempts"]) and bool(r["attempts"][0].get("ok")),
                       problems=[sorted({p_["problem"] for p_ in at.get("problems", [])}) for at in r["attempts"]],
                       errors=[at["error"] for at in r["attempts"] if at.get("error")],
                       chars=len(r.get("narrative") or ""), cost_usd=r["cost_usd"],
                       cached=r["usage"].get("prompt", 0) == 0, ms=round((time.perf_counter() - t0) * 1000),
                       text=r.get("narrative"), facts=r.get("facts"))
            rows.append(row)
            print(f"  #{p['alert_id']:<5} {rec['typology']:<28} 账户 {row['accounts']:>2} 交易 {row['txns']:>2} "
                  f"类别 {row['categories']}  理由 {r['status']:<9} 尝试 {row['attempts']}  "
                  f"{'首次通过' if row['first_pass'] else ('首次生成失败' if row['errors'] and not row['problems'][0] else '首次退回 ' + ','.join(row['problems'][0])) if row['attempts'] else ''}"
                  f"  {row['chars']} 字符  ${row['cost_usd']:.4f}")
        conn.rollback()                                            # 什么都不留下
    return {"split": split, "prompt_version": narr.PROMPT_VERSION, "builder": BUILDER, "analyst": ANALYST,
            "excluded": excluded, "rows": rows, "seconds": round(time.perf_counter() - t_all)}


def summarize(res: dict) -> None:
    rows = res["rows"]
    ok = [r for r in rows if r.get("draft") == "ok"]
    ver = [r for r in ok if r.get("narrative") == "verified"]
    print(f"\n[汇总] {res['split']}，{narr.PROMPT_VERSION}：案件 {len(rows)}（另排除 {len(res['excluded'])}）")
    print(f"  草稿组装并通过数据库检查：{len(ok)}/{len(rows)}")
    if ok:
        print(f"  理由第一次就通过核验：{sum(r['first_pass'] for r in ok)}/{len(ok)}；最终通过：{len(ver)}/{len(ok)}")
        from collections import Counter
        c = Counter(p for r in ok for at in r["problems"] for p in at)
        if c:
            print("  退回原因（按次数）：" + "，".join(f"{k} {v}" for k, v in c.most_common()))
        fresh = [r for r in ok if not r["cached"]]
        if ver:
            print(f"  理由长度：中位 {statistics.median(r['chars'] for r in ver):.0f} 字符"
                  f"（{min(r['chars'] for r in ver)}–{max(r['chars'] for r in ver)}）")
        if fresh:
            print(f"  费用（{len(fresh)} 条真实调用）：合计 ${sum(r['cost_usd'] for r in fresh):.3f}，"
                  f"每条中位 ${statistics.median(r['cost_usd'] for r in fresh):.4f}；"
                  f"耗时中位 {statistics.median(r['ms'] for r in fresh) / 1000:.0f}s")
        elif ok:
            print("  全部来自缓存回放：费用与耗时以第一次运行为准")


# ---------------------------------------------------------------------------
# 人工评分表
# ---------------------------------------------------------------------------
def write_rubric(res: dict) -> Path:
    ver = [r for r in res["rows"] if r.get("narrative") == "verified"]
    pick = sorted(random.Random(SEED).sample(ver, min(N_RUBRIC, len(ver))), key=lambda r: r["alert_id"])
    items = []
    for r in pick:
        src = r.get("sentence_sources") or []
        items.append({
            "alert_id": r["alert_id"], "typology": r["typology"],
            "facts": [f"step {f['step']} {f['path']} = {f['value']}" for f in r["facts"]],
            "sentences": [{"n": i, "text": s,
                           "numbers": [f"{x['number']} <- {', '.join(x['sources'])}" for x in
                                       (src[i - 1] if i - 1 < len(src) else [])],
                           "supported": None, "note": ""}
                          for i, s in enumerate(sentences(r["text"]), 1)],
            "flow_of_funds_clear": None, "why_suspicious_explained": None, "weak_signals_not_overstated": None,
            "comment": ""})
    doc = {"instructions": [
        "supported: true only if EVERY claim in the sentence follows from the evidence. 'numbers' shows where each "
        "number comes from, but a sentence can have correct numbers and still say too much.",
        "Mark supported: false when the sentence adds a conclusion that nothing measured, e.g. 'no funds were "
        "retained' (only one matched pair was checked), 'not explained by the business type' (no business type "
        "is recorded), 'purpose-built relationship', or a count read from a truncated list.",
        "The hardest cases have every number right: two different measurements joined into one story (e.g. the "
        "4 counterparties of all outflows described as where the matched pass-through funds went), or an "
        "observation from a limited window stated as the whole history ('no earlier activity' when only 7 days "
        "were checked). Mark these false too.",
        "Write the reason in note for every false.",
        "The three questions per item: true = good. flow_of_funds_clear: a reader can follow where the money came "
        "from and went. why_suspicious_explained: the text says why this is unusual, not only what happened. "
        "weak_signals_not_overstated: weak signals (first contact alone, one fan count, amount size) are not "
        "presented as proof.",
        "Do not change the text."], "split": res["split"], "prompt_version": res["prompt_version"],
        "seed": SEED, "items": items}
    p = rubric_path(res["split"])
    carried = 0
    if p.exists():
        # 人工评分花的是人的时间，不能被重跑覆盖：同一提示词版本、已经开始打分的评分表保留不动；
        # 版本不同的旧表改名存档（评分对应的是旧版本的理由）。新表里与旧表一字不差的句子、
        # 句子全部相同的整份理由，沿用旧表的判断（理由没变，判断也不该变），只有新出现的内容需要人去读
        old = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        started = any(s.get("supported") is not None for it in old.get("items", []) for s in it["sentences"])
        if started and old.get("prompt_version") == res["prompt_version"]:
            print(f"  评分表 {p.name} 已开始填写（{old['prompt_version']}），保留不动")
            return p
        if started:
            p.rename(p.with_name(f"{p.stem}_{old.get('prompt_version')}.yaml"))
            sent_old = {(it["alert_id"], s["text"]): s for it in old["items"] for s in it["sentences"]}
            item_old = {it["alert_id"]: it for it in old["items"]}
            for it in items:
                same = it["alert_id"] in item_old and [s["text"] for s in it["sentences"]] == \
                    [s["text"] for s in item_old[it["alert_id"]]["sentences"]]
                for s in it["sentences"]:
                    o = sent_old.get((it["alert_id"], s["text"]))
                    if o is not None and o.get("supported") is not None:
                        s["supported"], s["note"] = o["supported"], o.get("note", "")
                        carried += 1
                if same:
                    for k in ("flow_of_funds_clear", "why_suspicious_explained", "weak_signals_not_overstated",
                              "comment"):
                        if k in item_old[it["alert_id"]]:
                            it[k] = item_old[it["alert_id"]][k]
            if carried:
                print(f"  沿用旧评分表里一字不差的 {carried} 句的判断；新出现的句子需要重新阅读")
    p.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=110), encoding="utf-8")
    return p


def score(split: str, path: Path | None = None) -> None:
    doc = yaml.safe_load((path or rubric_path(split)).read_text(encoding="utf-8"))
    sent = [s for it in doc["items"] for s in it["sentences"]]
    # v2 的评分表用的是反向问题 weak_signal_as_strong（true = 有问题），容易填反；v3 起改成正向的
    # weak_signals_not_overstated（三项都是 true = 好）。读旧表时换算过来
    for it in doc["items"]:
        if "weak_signal_as_strong" in it and "weak_signals_not_overstated" not in it:
            v = it.pop("weak_signal_as_strong")
            it["weak_signals_not_overstated"] = None if v is None else not v
    keys = ("flow_of_funds_clear", "why_suspicious_explained", "weak_signals_not_overstated")
    todo = [s for s in sent if s["supported"] is None] + [it for it in doc["items"] for k in keys if it[k] is None]
    if todo:
        raise SystemExit(f"评分表还有 {len(todo)} 处没有填写")
    unsup = [s for s in sent if not s["supported"]]
    print(f"[人工评分] {split}，{doc['prompt_version']}，{len(doc['items'])} 份理由、{len(sent)} 句")
    print(f"  所引事实撑不起来的句子：{len(unsup)}/{len(sent)}（{len(unsup) / len(sent):.0%}）；"
          f"至少有一句的理由：{sum(any(not s['supported'] for s in it['sentences']) for it in doc['items'])}/"
          f"{len(doc['items'])}")
    for k, label in (("flow_of_funds_clear", "资金流向讲清楚"), ("why_suspicious_explained", "解释了为什么可疑"),
                     ("weak_signals_not_overstated", "没有把弱信号写成强证据")):
        print(f"  {label}：{sum(it[k] for it in doc['items'])}/{len(doc['items'])}")
    for it in doc["items"]:
        for s in it["sentences"]:
            if not s["supported"]:
                print(f"    #{it['alert_id']} 第 {s['n']} 句：{s['text'][:110]}  —— {s.get('note') or ''}")


def compare(split: str, other: Path) -> None:
    """两位评分者的一致率（D84）。other 只列与"全部为 true"不同的地方，未列出的视为 true。"""
    doc = yaml.safe_load(rubric_path(split).read_text(encoding="utf-8"))
    r2 = yaml.safe_load(other.read_text(encoding="utf-8"))
    s2 = {(x["alert_id"], x["n"]): x for x in r2.get("sentences", [])}
    i2 = {x["alert_id"]: x for x in r2.get("items", [])}
    keys = ("flow_of_funds_clear", "why_suspicious_explained", "weak_signals_not_overstated")
    n = agree = f1 = f2 = 0
    diff = []
    for it in doc["items"]:
        for s in it["sentences"]:
            a = bool(s["supported"])
            b = bool(s2.get((it["alert_id"], s["n"]), {}).get("supported", True))
            n += 1
            agree += a == b
            f1 += not a
            f2 += not b
            if a != b:
                diff.append((it["alert_id"], s["n"], a, b, s2.get((it["alert_id"], s["n"]), {}).get("note", ""),
                             s["text"]))
    print(f"[两位评分者] {split}：{n} 句，一致 {agree}（{agree / n:.0%}）；判为不被支撑：评分表 {f1} 句、"
          f"{r2.get('rater', '第二位')} {f2} 句"
          f"（其中临界 {sum(1 for x in r2.get('sentences', []) if x.get('borderline') and not x['supported'])} 句）")
    for k in keys:
        a = sum(bool(it[k]) for it in doc["items"])
        b = sum(bool(i2.get(it["alert_id"], {}).get(k, True)) for it in doc["items"])
        print(f"  {k}：评分表 {a}/{len(doc['items'])}，{r2.get('rater', '第二位')} {b}/{len(doc['items'])}")
    for aid, k, a, b, note, text in diff:
        print(f"  #{aid} 第 {k} 句  评分表 {a} / {r2.get('rater', '第二位')} {b}：{text[:90]}…\n      {note}")


def main():
    ap = argparse.ArgumentParser(description="R10d：STR 草稿的成批评测")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--score", action="store_true", help="汇总人工评分")
    ap.add_argument("--rubric", help="汇总指定的评分表（例如存档的旧版本）")
    ap.add_argument("--compare", help="与第二位评分者的评分比较（只列不同之处的 YAML）")
    a = ap.parse_args()
    if a.compare:
        compare(a.split, Path(a.compare))
        return
    if a.score:
        score(a.split, Path(a.rubric) if a.rubric else None)
        return
    if a.split == "test":
        if out_path("test").exists():
            raise SystemExit("reports/r10d_test.json 已存在：test 只跑一次")
        dev = json.loads(out_path("dev").read_text(encoding="utf-8")) if out_path("dev").exists() else None
        if dev is None or dev["prompt_version"] != narr.PROMPT_VERSION:
            raise SystemExit("test 必须用 dev 上评测过的同一个提示词版本：先在 dev 上运行当前版本")
    res = run(a.split)
    out_path(a.split).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    summarize(res)
    p = write_rubric(res)
    print(f"\n结果写到 {out_path(a.split).relative_to(PROJECT_ROOT)}；人工评分表 {p.relative_to(PROJECT_ROOT)}"
          f"（填好后运行 --split {a.split} --score）；评测数据已全部回滚")


if __name__ == "__main__":
    main()