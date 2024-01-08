src = """// any table from rdna3 docs"""

def h(v): return hex(int('0b'+'1'*(v+1), 2))
instructions = []
debug = ""
for line in src.splitlines():
  try:
    s = line.split(" ")
    if len(s) < 2: continue
    if not (s[1].startswith("[") and s[1].endswith("]")): continue
    instruction = s[0]
    r = s[1].replace("[", "").replace("]", "")
    end,start = r.split(":")
    end,start = int(end), int(start)
    if instruction == "ENCODING": continue
    instr = "instruction" if start == 0 else f"(instruction >> {start})"
    final = f"let {instruction.lower()} = {instr} & {h(end-start)};"
    instructions.append(instruction.lower())
    debug += f" {instruction.lower()}" + "={}"
    print(final)
  except: pass

debug = '"' + debug[1:] + '", '
for i in instructions:
  debug += f"{i}, "

print(debug)
