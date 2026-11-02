import csv
import json
import time
import urllib.request
import modal

app = modal.App("spec-bench")
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

DATA = "https://raw.githubusercontent.com/two-delta/DCOCO/main/data/{}.jsonl"
BENCHMARKS = ["gsm8k", "humaneval", "mt-bench"]

def vllm_image(*pkgs, precompiled=False):
    img = modal.Image.debian_slim(python_version="3.12").apt_install("git")
    if precompiled:
        img = img.env({"VLLM_USE_PRECOMPILED": "1"})
    return img.pip_install(*pkgs).env({"HF_HOME": "/cache"})

std_image = vllm_image("vllm==0.31.*", "flashmap[vllm]")
dflash2_image = vllm_image("vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/52816/head", precompiled=True)
dcoco_image = (
    modal.Image.from_dockerfile("DCOCO/docker/Dockerfile", context_dir="DCOCO/docker")
    .entrypoint([])
    .env({"HF_HOME": "/cache"})
)

RUNS = {
    "8b": ("Qwen/Qwen3-8B", {
        "dflash":   ("std",   {"method": "dflash", "model": "z-lab/Qwen3-8B-DFlash-b16", "num_speculative_tokens": 15}),
        "dcoco":    ("dcoco", {"method": "dflash", "model": "TD-Labs/Qwen3-8B-DCOCO", "num_speculative_tokens": 15}),
        "flashmap": ("std",   {"model": "j-shen/FlashMap-Qwen3-8B", "num_speculative_tokens": 16}),
    }),
    "4b": ("Qwen/Qwen3-4B", {
        "dflash":   ("std",     {"method": "dflash", "model": "z-lab/Qwen3-4B-DFlash-b16", "num_speculative_tokens": 15}),
        "dflash2":  ("dflash2", {"method": "dflash2", "model": "RedHatAI/Qwen3-4B-speculator.dflash2", "num_speculative_tokens": 8}),
        "dcoco":    ("dcoco",   {"method": "dflash", "model": "TD-Labs/Qwen3-4B-DCOCO", "num_speculative_tokens": 15}),
        "flashmap": ("std",     {"model": "j-shen/FlashMap-Qwen3-4B", "num_speculative_tokens": 16}),
    }),
}

def spec_counters(llm):
    m = {x.name: getattr(x, "value", 0) for x in llm.get_metrics()}
    return m.get("vllm:spec_decode_num_drafts", 0), m.get("vllm:spec_decode_num_accepted_tokens", 0)

def bench(target, spec, n, max_tokens):

    from vllm import LLM, SamplingParams
    llm = LLM(target, speculative_config=spec, max_model_len=8192, disable_log_stats=False)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0, max_tokens=max_tokens)

    def load(bench):
        lines = urllib.request.urlopen(DATA.format(bench)).read().decode().splitlines()[:n]
        return [
            tok.apply_chat_template([{"role": "user", "content": json.loads(l)["turns"][0]}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for l in lines
        ]

    results = {}
    for bench in BENCHMARKS:
        prompts = load(bench)
        llm.generate(prompts[:2], params, use_tqdm=False) 

        drafts0, accepted0 = spec_counters(llm)
        n_tokens, seconds = 0, 0.0
        for p in prompts:  # concurrence 1
            t0 = time.perf_counter()
            out = llm.generate([p], params, use_tqdm=False)[0]
            seconds += time.perf_counter() - t0
            n_tokens += len(out.outputs[0].token_ids)
        drafts, accepted = spec_counters(llm)

        gamma = 1 + (accepted - accepted0) / (drafts - drafts0) if drafts > drafts0 else 1.0
        results[bench] = {"tok_s": n_tokens / seconds, "gamma": gamma}
        print(f"{target} {spec and spec['model']} {bench}: {results[bench]}", flush=True)

    cache.commit()
    return results

GPU = dict(gpu="H200", volumes={"/cache": cache}, timeout=4 * 3600)

@app.function(image=std_image, **GPU)
def run_std(*args):
    return bench(*args)

@app.function(image=dflash2_image, **GPU)
def run_dflash2(*args):
    return bench(*args)

@app.function(image=dcoco_image, **GPU)
def run_dcoco(*args):
    return bench(*args)

RUNNERS = {"std": run_std, "dflash2": run_dflash2, "dcoco": run_dcoco}

@app.local_entrypoint()
def main(model: str = "8b", n: int = 80, max_tokens: int = 2048, k: int = 0):
    target, methods = RUNS[model]

    jobs = [("no-spec", image, None) for image in sorted({img for img, _ in methods.values()})]
    for name, (image, spec) in methods.items():
        if k:
            spec = {**spec, "num_speculative_tokens": k}
        jobs.append((name, image, spec))

    handles = [(name, image, RUNNERS[image].spawn(target, spec, n, max_tokens)) for name, image, spec in jobs]
    results = {(name, image): h.get() for name, image, h in handles}

    rows = []
    for (name, image), res in results.items():
        base = results[("no-spec", image)]
        for bench, r in res.items():
            rows.append({
                "model": model, "method": name, "image": image, "benchmark": bench,
                "gamma": round(r["gamma"], 2), "tok_s": round(r["tok_s"], 1),
                "speedup": round(r["tok_s"] / base[bench]["tok_s"], 2),
            })

    path = f"bench_{model}_k{k or 'natif'}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'méthode':<10}{'image':<9}{'benchmark':<11}{'γ':>6}{'tok/s':>9}{'speedup':>9}")
    for r in rows:
        print(f"{r['method']:<10}{r['image']:<9}{r['benchmark']:<11}{r['gamma']:>6}{r['tok_s']:>9}{r['speedup']:>9}")
    print(f"\n→ {path}")
