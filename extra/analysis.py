# prioritize which ops to implement
import os
import pandas as pd
import plotly.express as px

data = []
for base in ["./tests/test_ops", "./tests/test_dtype"]:
  files = os.listdir(base)
  for f in files:
    if not f.endswith(".s"): continue
    asm = open(base+"/"+f).read().splitlines()[6:]
    for i in asm:
      code = i.strip().split(" ")[0]
      data.append({ "test": f.split(".")[0], "code": code })

df = pd.DataFrame(data)
df = df.groupby("code").count().reset_index().sort_values(by="test", ascending=False)
fig = px.bar(df, x="code", y="test", title="frequency of each code based on number of tests").show()
