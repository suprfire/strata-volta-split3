"""Benchmark harness for a Strata Volta split serve.
Reads engine-side timings from /metrics; run from a machine that is NOT
the serving box so the box stays clean of unrelated traffic.
Paths: set HOST to your endpoint. Usage: python3 bench.py
"""
import json, time, random, threading, urllib.request

HOST = "http://<HOST>:8080"
MODEL = "qwen3.8-flash-next-split3"

BLOCK = "The quick brown fox catalogs quarterly fiscal projections across maritime timezones. "
NONCE = lambda: "Session id " + str(random.randint(100000, 999999)) + ". "

def post(prompt, max_tokens=256):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(HOST + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        json.loads(r.read())

def metrics():
    with urllib.request.urlopen(HOST + "/metrics", timeout=15) as r:
        return json.load(r)

def run_round(n, prompts, kind):
    before = {q["id"] for q in metrics()["requests"]}
    def w(i):
        post(prompts[i])
    ths = [threading.Thread(target=w, args=(i,)) for i in range(n)]
    t0 = time.time()
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.time() - t0
    recs = [q for q in metrics()["requests"] if q["id"] in set(metrics()["requests"]) - before if q["id"] not in before]
    return {"kind": f"{kind}-conc-{n}", "wall_s": round(wall, 2),
            "aggregate_out_tok_s": round(sum(q["output_tokens"] for q in recs) / wall, 2),
            "requests": [{"prompt_tokens": q["prompt_tokens"], "reused": q["reused"],
                          "output_tokens": q["output_tokens"],
                          "ttft_s": round(q["first_engine_token_s"], 3),
                          "prompt_ms": q["prompt_ms"], "decode_ms": q["decode_ms"],
                          "decode_tok_s": q["decode_tok_s"]} for q in recs]}

results = {"ts": time.strftime("%Y-%m-%d %H:%M %Z"), "model": MODEL, "rounds": []}
for chars in (6000, 24000, 96000):
    before = {q["id"] for q in metrics()["requests"]}
    post(NONCE() + BLOCK * (chars // len(BLOCK)), max_tokens=1)
    q = [x for x in metrics()["requests"] if x["id"] not in before][-1]
    results["rounds"].append({"kind": f"cold-prefill-{q['prompt_tokens']}tok",
                              "prompt_tokens": q["prompt_tokens"], "prompt_ms": q["prompt_ms"],
                              "prefill_tok_s": round(q["prompt_tokens"] * 1000 / q["prompt_ms"], 1)})
shared = BLOCK * 267
for n in (1, 2, 3, 4):
    results["rounds"].append(run_round(n, [NONCE() + shared + f" Unique {i}." for i in range(n)], "cold"))
    time.sleep(3)
    results["rounds"].append(run_round(n, [shared] * n, "warm"))
    time.sleep(3)
results["totals"] = metrics()["totals"]
results["engine"] = metrics()["engine"]
json.dump(results, open("results.json", "w"), indent=1)
print("BENCH-DONE ->", "results.json")
