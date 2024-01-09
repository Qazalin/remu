import json
import sys


asms = json.load(open("/tmp/asms.json"))
code = list(asms.keys())
print(asms[code[0]])
