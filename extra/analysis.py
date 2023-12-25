import os, string
import pandas as pd
import plotly.express as px

# prioritize which ops to implement
def code_freq(df):
  df = df.groupby("code").count().reset_index().sort_values(by="test", ascending=False)
  px.bar(df, x="code", y="test", title="frequency of each code based on number of tests").show()

def unique_binaries(df):
  df = df.loc[df["code"] == "s_mov_b32"]
  df = df.groupby("binary").count().reset_index()[["binary", "test"]].sort_values(by="test", ascending=False)
  px.bar(df, x="binary", y="test").show()
data = []
for base in ["./tests/test_ops", "./tests/test_dtype"]:
  files = os.listdir(base)
  for f in files:
    if not f.endswith(".s"): continue
    asm = open(base+"/"+f).read().splitlines()[6:]
    for i in asm:
      parts = i.strip().split(" ")
      code = parts[0]
      binary = ' '.join(s for s in parts[1:] if len(s) == 8 and all(c in string.hexdigits for c in s))
      data.append({ "test": f.split(".")[0], "code": code, "binary": binary, "line": i.strip() })

df = pd.DataFrame(data)
df = df.loc[df["code"] == "s_mov_b32"][["line", "binary"]]

def get_binary_at_idx(df):
  while True:
    try:
      i = input("idx: ")
      data = df[df.apply(lambda x: x["line"].startswith(f"s_mov_b32 {i}"), axis=1)]
      print(data["binary"].unique())
      if len(data["binary"].unique()) > 1:
        print(data["line"].unique())
    except:
      return


def lol(df):
  df = df[df.apply(lambda x: x["line"].startswith(f"s_mov_b32"), axis=1)]
  s_vals = []
  for _,data in df.iterrows(): print(' '.join(data["line"].strip().split(" ")[1:3]).split(", "))

get_binary_at_idx(df)
