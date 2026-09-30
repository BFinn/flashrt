#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""sw115: a code-editing teacher text. A chat request asks for a source file back with two
identifiers renamed; the answer is that file with the renames. Writes the token ids of the
rendered prompt followed by the answer's, and prints the prompt's length.
Usage: make_code_edit.py SERVER MODEL SOURCE OUT_IDS"""
import json
import os
import subprocess
import sys
import tempfile

server, model, src, out = sys.argv[1:5]
code = open(src).read()
renames = [("SysFlag", "MailboxFlag"), ("kMaxWindow", "kMaxVerifyWindow")]
answer = code
for a, b in renames:
    answer = answer.replace(a, b)
ask = (f"Rename `{renames[0][0]}` to `{renames[0][1]}` and `{renames[1][0]}` to `{renames[1][1]}` in this file. "
       "Output the whole file and nothing else.\n\n```cpp\n" + code + "```")
with tempfile.TemporaryDirectory() as d:
    req = os.path.join(d, "req.json")
    json.dump({"messages": [{"role": "user", "content": ask}], "chat_template_kwargs": {"enable_thinking": False}}, open(req, "w"))
    rendered = subprocess.run([server, "--model", model, "--render", req], check=True, capture_output=True, text=True).stdout
    prompt = rendered[: rendered.rindex("\n--- ")]   # --render prints the prompt, a newline, a count line
    tp = os.path.join(d, "p.txt")
    ta = os.path.join(d, "a.txt")
    open(tp, "w").write(prompt)
    open(ta, "w").write("```cpp\n" + answer + "```<|im_end|>")
    tok = lambda p: subprocess.run([server, "--model", model, "--tokenize", p], check=True, capture_output=True, text=True).stdout.split()
    pids, aids = tok(tp), tok(ta)
open(out, "w").write(" ".join(pids + aids) + "\n")
print(len(pids), len(aids))
