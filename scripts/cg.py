import pandas as pd
amt = pd.read_csv("data/raw/HI-Small_Trans.csv",
                  usecols=["Amount Paid", "Amount Received"],
                  dtype=str)                       # 关键：按原始字符串读，不转 float
s = pd.concat([amt["Amount Paid"], amt["Amount Received"]])
dec = s.str.extract(r"\.(\d+)$")[0].fillna("").str.len()
print("最大小数位:", dec.max())
print("小数位分布:\n", dec.value_counts().sort_index())
print("含科学计数法的行数:", s.str.contains("[eE]").sum())
print("最小的5个金额:", sorted(s.unique())[:5])