"""
SentryAML - R9c-0：分诊评测的切分与抽样（在 Agent 第一次跑真实告警之前完成，D72）
================================================================================
两个阶段，规则在 D72 里预先写定：

  --inspect   只看结构、不读标签：按两种"连接"定义算告警的连通分量，打印分量大小分布。
              用 D72 写定的规则选定其中一种（不是看着结果挑）。
  --split     用选定的定义：排除看过的告警所在分量 → 分量随机分到 dev / test → 每边按（档位 × 是否洗钱）
              分层抽样，每个分量最多抽 1 条（最稀少的层先抽）→ 写出 reports/r9c_split.json（test 一经写出即冻结）。

为什么按连通分量切（D68 设计原则六）：同一伙人的告警往往共享账户。如果一条落在 dev、另一条落在 test，
  在 dev 上调提示词时看到的那伙人的账户、资金路径，会在 test 上被"认出来"，测试结果偏乐观。
  把通过共享账户连在一起的告警作为一个整体，整体只进一边。

两种连接定义：
  L1  两条告警共享一个账户（付款方或收款方）就连在一起
  L2  在 L1 的基础上，两条告警的账户之间在 9/1–9/11 有过直接交易，也连在一起（同一团伙的告警不一定共享账户，
      但团伙内部通常互相转账）
  两种定义都不经过清算层账户：清算层是基础设施（D17），一个清算层账户连着几千个毫不相干的对手方，
  如果允许经过它连接，几乎所有告警都会并成一个分量。

用法：
    python scripts/r9c_split.py --inspect
    python scripts/r9c_split.py --split            # 规则写定、--inspect 看过分量大小之后
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "reports" / "r9c_split.json"
SEED = 20261008
# D72：每边的抽样配额（档位 × 是否洗钱）
QUOTA = {("T1", 1): 10, ("T1", 0): 10, ("T2", 1): 20, ("T2", 0): 20}
# D72：L2 的最大分量不超过全部告警的 10% 时用 L2，否则用 L1（在看分量大小之前写定）
L2_MAX_SHARE = 0.10


class DSU:
    """并查集：把"连在一起"的告警合并成分量。"""
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def load_alerts(cur) -> list[dict]:
    cur.execute("""SELECT a.alert_id, a.tier, a.score, a.from_bank, a.from_account, a.to_bank, a.to_account,
                          COALESCE(df.total_cp_count, 0) > 200 AS payer_clearing,
                          COALESCE(dt.total_cp_count, 0) > 200 AS payee_clearing
                   FROM alerts a
                   LEFT JOIN account_degrees df ON df.bank_id = a.from_bank AND df.account_number = a.from_account
                   LEFT JOIN account_degrees dt ON dt.bank_id = a.to_bank AND dt.account_number = a.to_account
                   ORDER BY a.alert_id""")
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def components(alerts: list[dict], links: str, cur) -> dict[int, int]:
    """返回 alert_id → 分量编号（分量里最小的 alert_id）。"""
    dsu = DSU([a["alert_id"] for a in alerts])
    by_acct: dict[tuple, list[int]] = defaultdict(list)
    for a in alerts:
        if not a["payer_clearing"]:
            by_acct[(a["from_bank"], a["from_account"])].append(a["alert_id"])
        if not a["payee_clearing"]:
            by_acct[(a["to_bank"], a["to_account"])].append(a["alert_id"])
    for ids in by_acct.values():                       # L1：共享账户
        for x in ids[1:]:
            dsu.union(ids[0], x)
    if links == "L2":                                  # L2：告警账户之间的直接交易
        accts = list(by_acct)
        cur.execute("CREATE TEMP TABLE r9c_acct (bank INT, acct TEXT) ON COMMIT DROP")
        with cur.copy("COPY r9c_acct (bank, acct) FROM STDIN") as cp:
            for b, a in accts:
                cp.write_row((b, a))
        cur.execute("""SELECT DISTINCT t.from_bank, t.from_account::text, t.to_bank, t.to_account::text
                       FROM txn_norm t
                       JOIN r9c_acct x ON x.bank = t.from_bank AND x.acct = t.from_account
                       JOIN r9c_acct y ON y.bank = t.to_bank AND y.acct = t.to_account
                       WHERE t.txn_ts >= '2022-09-01' AND t.txn_ts < '2022-09-12'
                         AND NOT (t.from_bank = t.to_bank AND t.from_account = t.to_account)""")
        for fb, fa, tb, ta in cur.fetchall():
            dsu.union(by_acct[(fb, fa)][0], by_acct[(tb, ta)][0])
    return {a["alert_id"]: dsu.find(a["alert_id"]) for a in alerts}


def size_report(comp: dict[int, int], n: int, name: str) -> float:
    sizes = Counter(comp.values())
    dist = sorted(sizes.values(), reverse=True)
    buckets = Counter("1" if s == 1 else "2-5" if s <= 5 else "6-20" if s <= 20 else ">20" for s in dist)
    share = dist[0] / n
    print(f"  {name}：{len(dist):,} 个分量；最大的 {dist[0]} 条告警（占 {share:.1%}），前 5 大 {dist[:5]}；"
          f"大小分布 1 条 {buckets['1']}、2–5 条 {buckets['2-5']}、6–20 条 {buckets['6-20']}、>20 条 {buckets['>20']}")
    return share


def excluded_ids() -> set[int]:
    """看过标签或看过工具输出的告警：R9a 冒烟测试（D68）+ R9b-2 冒烟测试里的清算层告警（D71）。"""
    shown = json.loads((PROJECT_ROOT / "reports" / "r9a_smoke_alerts.json").read_text(encoding="utf-8"))["alert_ids"]
    return set(shown) | {696}


def inspect():
    with connect() as conn, conn.cursor() as cur:
        alerts = load_alerts(cur)
        n = len(alerts)
        print(f"告警 {n:,} 条；付款方是清算层 {sum(a['payer_clearing'] for a in alerts)} 条，"
              f"收款方是清算层 {sum(a['payee_clearing'] for a in alerts)} 条（不经过它们连接）\n")
        s1 = size_report(components(alerts, "L1", cur), n, "L1 共享账户")
        s2 = size_report(components(alerts, "L2", cur), n, "L2 共享账户 + 直接交易")
    choice = "L2" if s2 <= L2_MAX_SHARE else "L1"
    print(f"\n按 D72 的规则（L2 最大分量 ≤ {L2_MAX_SHARE:.0%} 则用 L2）：选定 {choice}"
          + ("" if choice == "L2" else f"（L2 最大分量占 {s2:.1%}，会让一边独占大部分告警）"))
    return choice


def split():
    if OUT.exists():
        raise SystemExit(f"{OUT.relative_to(PROJECT_ROOT)} 已存在：切分一经写出即冻结，不重新抽样")
    choice = inspect()
    with connect() as conn, conn.cursor() as cur:
        alerts = load_alerts(cur)
        comp = components(alerts, choice, cur)
        cur.execute("SELECT alert_id, is_laundering FROM alert_labels")
        label = dict(cur.fetchall())                    # 只用于分层抽样，不出现在任何 Agent 的输入里
    ex = excluded_ids()
    bad_comps = {comp[i] for i in ex if i in comp}
    pool = [a for a in alerts if comp[a["alert_id"]] not in bad_comps]
    print(f"\n排除看过的告警 {sorted(ex)} 所在的 {len(bad_comps)} 个分量，"
          f"共 {len(alerts) - len(pool)} 条告警；剩余 {len(pool):,} 条")

    rng = random.Random(SEED)
    comp_ids = sorted({comp[a["alert_id"]] for a in pool})
    rng.shuffle(comp_ids)
    side = {c: ("dev" if i % 2 == 0 else "test") for i, c in enumerate(comp_ids)}   # 分量交替分到两边

    out = {"seed": SEED, "links": choice, "quota": {f"{t}/{l}": q for (t, l), q in QUOTA.items()},
           "excluded_alerts": sorted(ex), "dev": [], "test": [], "strata_population": {}, "frozen": "test"}
    # 每边：按层抽样，每个分量最多抽 1 条。最稀少的层先抽（T2 洗钱只有 174 条），避免它的分量被别的层先占用
    order = sorted(QUOTA, key=lambda k: sum(1 for a in pool if (a["tier"], label[a["alert_id"]]) == k))
    for s in ("dev", "test"):
        used: set[int] = set()
        n_comp = len({comp[a["alert_id"]] for a in pool if side[comp[a["alert_id"]]] == s})
        print(f"\n  {s}：{n_comp:,} 个分量")
        for key in order:
            by_comp = defaultdict(list)
            for a in pool:
                c = comp[a["alert_id"]]
                if side[c] == s and c not in used and (a["tier"], label[a["alert_id"]]) == key:
                    by_comp[c].append(a)
            cids = sorted(by_comp)
            rng.shuffle(cids)
            q = QUOTA[key]
            for c in cids[:q]:
                a = rng.choice(by_comp[c])
                used.add(c)
                # m = 这个分量里同一层的告警数。先等概率抽分量、再在分量里抽 1 条，所以大分量里的告警被抽中的概率
                # 是小分量的 1/m；R9c 把样本加权回总体时，每条的权重要乘以 m（两阶段抽样的标准做法，D72）
                out[s].append({"alert_id": a["alert_id"], "tier": key[0],
                               "stratum": f"{key[0]}/{'L' if key[1] else 'N'}", "component": c,
                               "m": len(by_comp[c])})
            print(f"    {key[0]} {'洗钱' if key[1] else '非洗钱'}：含这一层告警的可用分量 {len(cids):>5}，"
                  f"抽取 {min(q, len(cids))}" + ("  ← 不足配额" if len(cids) < q else ""))
    # 总体规模（全部 4,226 条，按层计数）：R9c 用它把样本里各层的比率加权回到整个告警队列
    for a in alerts:
        k = f"{a['tier']}/{'L' if label[a['alert_id']] else 'N'}"
        out["strata_population"][k] = out["strata_population"].get(k, 0) + 1
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    dev_c, test_c = {x["component"] for x in out["dev"]}, {x["component"] for x in out["test"]}
    print(f"\n  dev {len(out['dev'])} 条、test {len(out['test'])} 条；两边共享的分量 {len(dev_c & test_c)} 个（应为 0）")
    print(f"  总体各层规模：{out['strata_population']}")
    print(f"  已写到 {OUT.relative_to(PROJECT_ROOT)}；test 冻结：在评测规则写定、只跑一次之前，不在 test 告警上运行 Agent")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inspect", action="store_true")
    g.add_argument("--split", action="store_true")
    a = ap.parse_args()
    inspect() if a.inspect else split()