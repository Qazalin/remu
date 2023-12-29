import numpy as np
import os, string
import pandas as pd
import plotly.express as px

# prioritize which ops to implement
def code_freq(df):
  df = df.groupby("code").count().reset_index().sort_values(by="test", ascending=False)
  px.bar(df, x="code", y="test", title="frequency of each code based on number of tests").show()
  #px.pie(df, values="test", names="code", title="frequency of each code based on number of tests").show()

def unique_binaries(df):
  df = df.loc[df["code"] == "s_mov_b32"]
  df = df.groupby("hex").count().reset_index()[["hex", "test"]].sort_values(by="test", ascending=False)
  px.bar(df, x="hex", y="test").show()
data = []
for base in ["./tests/test_ops", "./tests/test_dtype"]:
  files = os.listdir(base)
  for f in files:
    if not f.endswith(".s"): continue
    asm = open(base+"/"+f).read().splitlines()[6:]
    for i in asm:
      parts = i.strip().split(" ")
      code = parts[0]
      hexval = ' '.join(s for s in parts[1:] if len(s) == 8 and all(c in string.hexdigits for c in s))
      data.append({ "test": f.split(".")[0], "code": code, "hex": hexval, "line": i.strip() })

df = pd.DataFrame(data)
df["instruction0"] = df.apply(lambda x: int("0x" + x["hex"].split(" ")[0], 16), axis=1)
df["instruction1"] = df.apply(lambda x: int("0x" + x["hex"].split(" ")[1], 16) if len(x["hex"].split(" ")) > 1 else 0, axis=1)

def get_binary_at_idx(df):
  v = "v_add_f32_e32"
  df = df.loc[df["code"] == v][["line", "hex"]]
  df.drop_duplicates(inplace=True)
  print(df.head())
  while True:
    try:
      i = input("idx: ")
      data = df[df.apply(lambda x: x["line"].startswith(f"{v} {i}"), axis=1)]
      print(data["hex"].unique())
      print(data["line"].unique())
    except:
      return
# sop2: last two bits are 0b10 and the opcode bits ((instruction0 >> 23) & 0xFF) are between 0-53
sop2 = df[df.apply(lambda x: ((x["instruction0"] >> 30) == 0b10) and ((x["instruction0"] >> 23) & 0xFF) <= 53, axis=1)]

# smem: startswith 111101 and is 64 bits long (two instructions)
smem = df[df.apply(lambda x: x["instruction0"] >> 26 == 0b111101 and x["instruction1"] != 0, axis=1)]
smem = smem[smem["code"] == "s_load_b32"]
smem = smem[["line", "instruction0", "instruction1", "hex"]]
smem = smem.groupby(["line", "instruction0", "instruction1", "hex"]).count().reset_index()
smem["line"] = smem["line"].apply(lambda x: x.split("/")[0].strip())
smem["hex"] = smem["hex"].apply(lambda x: "0x"+x.split(" ")[0].lower() + ", " + "0x"+x.split(" ")[1].lower())
# select where smem["line"] has a - char in it
smem = smem[smem.apply(lambda x: "-" in x["line"], axis=1)]
smem["offset"] = smem["instruction1"] & 0x1fffff
smem["offset_signed"] = smem["offset"].apply(lambda x: x - 0x200000 if x > 0x100000 else x)
smem["instruction1"] = smem["instruction1"].apply(lambda x: bin(x))
print(smem)

sop1 = df[df.apply(lambda x: x["instruction0"] >> 23 == 0b10_1111101, axis=1)]
#code_freq(sop1)
#print(sop1.loc[sop1["line"].str.startswith("s_mov_b64")].head()["line"].unique())


vop2 = df[df.apply(lambda x: x["instruction0"] >> 31 == 0b0 and not (x["instruction0"] >> 25 == 0b0111111), axis=1)]
#print(vop2.head())
#code_freq(vop2)
