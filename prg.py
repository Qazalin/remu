import json
import sys


asms = json.load(open("/tmp/asms.json"))
idx = int(sys.argv[1])
v = sys.argv[2]

code = list(asms.keys())[idx]
if v == "c":
  print(code)
else:
  print(asms[list(asms.keys())[idx]])
