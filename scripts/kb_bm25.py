"""
SentryAML - RAG 层 R5b：BM25 稀疏向量
=====================================
BM25 给"问题 q 和块 d"打分：

    score(q, d) = Σ_{词 t 同时在 q 和 d 中}  IDF(t) × tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )

  - IDF(t)：词越稀有权重越高。"COSMIC" 只出现在少数块里，命中它很说明问题；"bank" 到处都是，几乎不加分
  - tf 部分：出现次数越多越相关，但会饱和（k1 控制饱和速度），出现 10 次不等于 2 次的 5 倍
  - 长度校正：长块天然词多，按 |d|/avgdl 打折（b 控制力度）

分工：
  - 本文件算"tf 部分"，把每个块变成稀疏向量 {词的编号: 权重}
  - IDF 依赖整个集合（有多少块含这个词），由 Qdrant 在服务端统计（集合设置 modifier=IDF），
    增删文档时自动更新，不用我们维护
  - 查询时每个词权重为 1，Qdrant 再乘上 IDF，与文档权重相乘求和，就得到上面的公式

为什么自己写这几十行，而不用 fastembed 的 "Qdrant/bm25"：
  - fastembed 要从 Hugging Face 下载配置文件并依赖 onnxruntime，对一个公式来说太重
  - 自己写可以针对监管文本定制分词：保留 "6.14C"、"14-1"、"u-turn"、"ML/TF" 这类编号和复合词
  - 完全透明、可单元测试，面试时能讲清每一步

为什么用哈希当词的编号，而不维护词表：查询时不需要加载词表文件，新词也有编号；
31 位哈希在几万个词的规模下几乎不会冲突，偶发冲突只会让两个词共享一个权重。
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

import snowballstemmer

K1, B = 1.2, 0.75
_stem = snowballstemmer.stemmer("english").stemWord

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves also may must shall within without upon any etc e.g i.e via
""".split())

# 一个词：字母数字，允许中间用 . - / 连接（保留条款编号 6.14C、14-1，复合词 pass-through、ML/TF）
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "―": "-"}))
    text = re.sub(r"\[\^\d+\]", " ", text)                     # 脚注标记不是内容
    out = []
    for tok in TOKEN_RE.findall(text):
        tok = tok.strip(".-/")
        if not tok:
            continue
        if re.search(r"[.\-/]", tok):
            out.append(tok)                                    # 复合词/编号整体保留
            out += [_stem(p) for p in re.split(r"[.\-/]", tok)  # 其中的字母部分也单独计入
                    if p.isalpha() and len(p) > 1 and p not in STOPWORDS]
        elif tok in STOPWORDS or (len(tok) == 1 and tok.isalpha()):
            continue
        else:
            out.append(_stem(tok) if tok.isalpha() else tok)
    return out


def term_id(tok: str) -> int:
    return int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def avg_len(texts: list[str]) -> float:
    return sum(len(tokenize(t)) for t in texts) / max(len(texts), 1)


def doc_vector(text: str, avgdl: float) -> tuple[list[int], list[float]]:
    toks = tokenize(text)
    dl = len(toks)
    tf = Counter(term_id(t) for t in toks)
    norm = K1 * (1 - B + B * dl / avgdl)
    idx = sorted(tf)
    return idx, [tf[i] * (K1 + 1) / (tf[i] + norm) for i in idx]


def query_vector(text: str) -> tuple[list[int], list[float]]:
    idx = sorted({term_id(t) for t in tokenize(text)})
    return idx, [1.0] * len(idx)


if __name__ == "__main__":
    for s in ["Paragraph 6.14C of the Notice: beneficial owner's date of birth",
              "Funds received are immediately remitted — “U-turn” transactions (ML/TF risk), pass-through payments",
              "账户收到大额转账后马上转走"]:
        print(s, "\n  →", tokenize(s))