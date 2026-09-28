import json, sys, time, urllib.request
url = sys.argv[1]
text = open("$BENCH/p0c-20260927/wiki.txt").read()
def run(prompt, n=128, temp=1.0):
    body = {"prompt": prompt, "max_tokens": n, "temperature": temp, "top_p": 0.95, "top_k": 20, "seed": 1, "stream": True}
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; k = 0; out = ""
    for raw in urllib.request.urlopen(req, timeout=3600):
        l = raw.decode().strip()
        if not l.startswith("data:") or l == "data: [DONE]": continue
        d = json.loads(l[5:])
        if d["choices"][0]["text"]:
            first = first or time.time(); k += 1; out += d["choices"][0]["text"]
    dt = time.time() - (first or time.time())
    print(f"first text after {(first or time.time()) - t0:.1f} s, {k} chunks in {dt:.2f} s ({k / max(dt, 1e-9):.1f}/s): {out[:60]!r}", flush=True)
    return out
for chars in (9000, 140000):
    p = text[:chars]; p = p[:p.rfind(". ") + 1]
    o = run(p)
    run(p + o + " And furthermore,", 64)   # continuation: reuse
