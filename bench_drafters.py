import csv
import json
import subprocess
import time
import urllib.request
from pathlib import Path
import modal
import os

app = modal.App("spec-bench")
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

BENCHMARKS = ["gsm8k", "humaneval", "mt-bench"]

ENV = {"HF_HOME": "/cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
PYTHON = "RUN ln -sf $(command -v python3) /usr/local/bin/python"
BENCH_DEPS = ["pandas", "datasets", "matplotlib", "seaborn", "scipy"] 

def vllm_base():
    return (
        modal.Image.from_registry("vllm/vllm-openai:v0.31.0")
        .dockerfile_commands([PYTHON])
        .entrypoint([])
        .env(ENV)
        .pip_install(*BENCH_DEPS)
    )

std_image = vllm_base().pip_install("flashmap[vllm]").add_local_dir("DCOCO", "/root/DCOCO")
dcoco_image = (
    modal.Image.from_dockerfile("DCOCO/docker/Dockerfile", context_dir="DCOCO/docker")
    .dockerfile_commands([PYTHON])
    .entrypoint([])
    .env(ENV)
    .add_local_dir("DCOCO", "/root/DCOCO")
)

RUNS = {
    "8b": ("Qwen/Qwen3-8B", {
        "dflash":   ("std",   {"method": "dflash", "model": "z-lab/Qwen3-8B-DFlash-b16", "num_speculative_tokens": 15}),
        "dcoco":    ("dcoco", {"method": "dflash", "model": "TD-Labs/Qwen3-8B-DCOCO", "num_speculative_tokens": 15}),
        "flashmap": ("std",   {"model": "j-shen/FlashMap-Qwen3-8B", "num_speculative_tokens": 16}),
        "flashmap2": ("std",  {"model": "j-shen/FlashMap-Qwen3-8B", "num_speculative_tokens": 16, "_env": {"FLASHMAP_NUM_STEPS": "2"}}),
    }),
    "4b": ("Qwen/Qwen3-4B", {
        "dflash":   ("std",     {"method": "dflash", "model": "z-lab/Qwen3-4B-DFlash-b16", "num_speculative_tokens": 15}),
        "dcoco":    ("dcoco",   {"method": "dflash", "model": "TD-Labs/Qwen3-4B-DCOCO", "num_speculative_tokens": 15}),
        "flashmap": ("std",     {"model": "j-shen/FlashMap-Qwen3-4B", "num_speculative_tokens": 16}),
        "flashmap2": ("std",  {"model": "j-shen/FlashMap-Qwen3-4B", "num_speculative_tokens": 16, "_env": {"FLASHMAP_NUM_STEPS": "2"}}),
    }),
}

def wait_until_up(url, server, timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        if server.poll() is not None:
            raise RuntimeError(f"le serveur vLLM s'est arrêté (code {server.returncode}), voir les logs au-dessus")
        try:
            urllib.request.urlopen(url)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError("le serveur vLLM n'a pas démarré")

def parse_table(text):
    results = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0] in BENCHMARKS and cells[1]:
            results[cells[0]] = {
                "tok_s": float(cells[1].split()[0]),
                "gamma": float(cells[3].split()[0]) if cells[3] else 1.0,
            }
    return results

def bench(target, spec, rows):
    spec = dict(spec or {})
    env = {**os.environ, **spec.pop("_env", {})}  
    cmd = ["vllm", "serve", target, "--max-model-len", "8192", "--port", "8000"]
    if spec:
        cmd += ["--speculative-config", json.dumps(spec)]
    server = subprocess.Popen(cmd, env=env)
    wait_until_up("http://localhost:8000/health", server)

    out = subprocess.run(
        ["python", "/root/DCOCO/scripts/benchmark.py", "--endpoint", "http://localhost:8000", "--out", "/tmp/out",
         "--datasets", *BENCHMARKS, "--configs", "t0", "--rows", str(rows), "--vllm-bin", "vllm"],
        capture_output=True, text=True,
    )
    print(out.stdout, out.stderr, flush=True)
    for log in Path("/tmp/out").rglob("*.log"):
        print(f"--- {log} ---\n" + "\n".join(log.read_text().splitlines()[-30:]), flush=True)
    server.terminate()
    cache.commit()
    return parse_table(out.stdout + out.stderr)

GPU = dict(gpu="H200", volumes={"/cache": cache}, timeout=4 * 3600)

@app.function(image=std_image, **GPU)
def run_std(*args):
    return bench(*args)

@app.function(image=dcoco_image, **GPU)
def run_dcoco(*args):
    return bench(*args)

RUNNERS = {"std": run_std, "dcoco": run_dcoco}

@app.local_entrypoint()
def main(model: str = "8b", rows: int = 80, k: int = 0, only: str = ""):
    target, methods = RUNS[model]
    if only:
        methods = {m: v for m, v in methods.items() if m in only.split(",")}

    jobs = [("no-spec", image, None) for image in sorted({img for img, _ in methods.values()})]
    for name, (image, spec) in methods.items():
        if k:
            spec = {**spec, "num_speculative_tokens": k}
        jobs.append((name, image, spec))

    handles = [(name, image, RUNNERS[image].spawn(target, spec, rows)) for name, image, spec in jobs]
    results = {}
    for name, image, h in handles:
        try:
            results[(name, image)] = h.get()
        except Exception as e:
            print(f"\n✗ {name} ({image}) a échoué : {e}\n")

    table = []
    for (name, image), res in results.items():
        base = results.get(("no-spec", image), {})
        for bench_name, r in res.items():
            if bench_name in base:
                table.append({
                    "model": model, "method": name, "image": image, "benchmark": bench_name,
                    "gamma": round(r["gamma"], 2), "tok_s": round(r["tok_s"], 1),
                    "speedup": round(r["tok_s"] / base[bench_name]["tok_s"], 2),
                })

    if not table:
        print("aucun résultat")
        return

    path = f"bench_{model}_k{k or 'natif'}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table[0].keys())
        w.writeheader()
        w.writerows(table)

    print(f"\n{'méthode':<10}{'image':<9}{'benchmark':<11}{'γ':>6}{'tok/s':>9}{'speedup':>9}")
    for r in table:
        print(f"{r['method']:<10}{r['image']:<9}{r['benchmark']:<11}{r['gamma']:>6}{r['tok_s']:>9}{r['speedup']:>9}")
    print(f"\n→ {path}")