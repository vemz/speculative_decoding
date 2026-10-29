import json
import random
import modal

app = modal.App("flashmap-vs-dflash")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
base = modal.Image.debian_slim(python_version="3.12").env({"HF_HOME": "/cache"})
flashmap_image = base.pip_install("flashmap==0.1.0", "transformers>=5.16,<5.21", "torch>=2.9,<2.14")
dflash_image = base.pip_install("dflash[local]==0.1.0")  # versions fixées par le dépôt DFlash

TARGET = "Qwen/Qwen3-8B"
FLASHMAP = "j-shen/FlashMap-Qwen3-8B"
DFLASH = "z-lab/Qwen3-8B-DFlash-b16"
DATA = "https://raw.githubusercontent.com/deepseek-ai/DeepSpec/main/eval_datasets/{}.jsonl"
BENCHMARKS = ["gsm8k", "humaneval", "mt-bench"]
GPU = dict(gpu="H100", volumes={"/cache": hf_cache}, timeout=3 * 3600)

CONFIGS = {
    "FlashMap 1 pas": ("flashmap", 1, 16),
    "FlashMap 2 pas": ("flashmap", 2, 16),
    "FlashMap 1 pas, K=15": ("flashmap", 1, 15),
    "DFlash b16": ("dflash", None, 15),
}
PAPER = { 
    "gsm8k": {"FlashMap 1 pas": 7.69, "FlashMap 2 pas": 8.31, "DFlash b16": 7.43},
    "humaneval": {"FlashMap 1 pas": 5.82, "FlashMap 2 pas": 6.26, "DFlash b16": 5.74},
    "mt-bench": {"FlashMap 1 pas": 3.34, "FlashMap 2 pas": 3.40, "DFlash b16": 3.14},
}

def load_prompts(bench, num_prompts):
    import urllib.request

    lines = urllib.request.urlopen(DATA.format(bench)).read().decode().splitlines()
    return [json.loads(line)["turns"][0] for line in lines[:num_prompts]]  # premier tour seulement

def encode(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt", return_dict=True,
    )["input_ids"].cuda()

@app.function(image=flashmap_image, **GPU)
def prefetch():
    from huggingface_hub import snapshot_download

    for repo in (TARGET, FLASHMAP, DFLASH):
        snapshot_download(repo)
    hf_cache.commit()

@app.function(image=flashmap_image, **GPU)
def run_flashmap(bench, num_prompts, max_new_tokens, steps, k):
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16).cuda().eval()
    draft = AutoModel.from_pretrained(FLASHMAP, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    draft.config.block_size = k  # tokens draftés par tour

    prompts = load_prompts(bench, num_prompts)
    draft.spec_generate(target, encode(tokenizer, prompts[0]), max_new_tokens=32, num_steps=steps)  # échauffement
    rows = []
    for prompt in prompts:
        out = draft.spec_generate(target, encode(tokenizer, prompt), max_new_tokens=max_new_tokens,
                                  temperature=0.0, num_steps=steps)
        rows.append((sum(out.acceptance_lengths), len(out.acceptance_lengths)))
    return rows

@app.function(image=dflash_image, **GPU)
def run_dflash(bench, num_prompts, max_new_tokens):
    from dflash.benchmark import load_transformers_models, stop_token_ids
    from dflash.model import dflash_generate

    target, draft, tokenizer = load_transformers_models(TARGET, DFLASH, "cuda")
    stops = stop_token_ids(target, tokenizer)

    prompts = load_prompts(bench, num_prompts)
    dflash_generate(draft, target, encode(tokenizer, prompts[0]), 32, stops)  # échauffement
    rows = []
    for prompt in prompts:
        out = dflash_generate(draft, target, encode(tokenizer, prompt), max_new_tokens, stops,
                              temperature=0.0, return_stats=True)
        rows.append((sum(out.acceptance_lengths), len(out.acceptance_lengths)))
    return rows

def gamma(rows, idx=None):
    """γ = total des tokens validés / total des tours (comme les benchmarks officiels)."""
    idx = range(len(rows)) if idx is None else idx
    return sum(rows[i][0] for i in idx) / sum(rows[i][1] for i in idx)

def bootstrap(stat, n, b=2000, seed=0):
    """Intervalle à 95 % d'une statistique, en rééchantillonnant les prompts."""
    rng = random.Random(seed)
    values = sorted(stat(rng.choices(range(n), k=n)) for _ in range(b))
    return values[int(0.025 * b)], values[int(0.975 * b)]

def summarize(results):
    print(f"\n{'benchmark':<11}{'drafter':<23}{'γ':>6}{'IC 95 %':>16}{'papier':>9}")
    for bench, by_config in results.items():
        for name, rows in by_config.items():
            lo, hi = bootstrap(lambda idx: gamma(rows, idx), len(rows))
            paper = PAPER[bench].get(name)
            print(f"{bench:<11}{name:<23}{gamma(rows):>6.2f}{f'[{lo:.2f}, {hi:.2f}]':>16}"
                  f"{'' if paper is None else f'{paper:.2f}':>9}")

    print("\nÉcart avec DFlash, mêmes prompts (bootstrap apparié) :")
    for bench, by_config in results.items():
        ref = by_config["DFlash b16"]
        for name in ("FlashMap 1 pas", "FlashMap 1 pas, K=15", "FlashMap 2 pas"):
            rows = by_config[name]
            diff = lambda idx: gamma(rows, idx) - gamma(ref, idx)
            lo, hi = bootstrap(diff, len(rows))
            verdict = "significatif" if lo > 0 or hi < 0 else "non significatif"
            print(f"  {bench:<11}{name:<23}{diff(range(len(rows))):+.2f}  [{lo:+.2f}, {hi:+.2f}]  {verdict}")

@app.local_entrypoint()
def main(num_prompts: int = 164, max_new_tokens: int = 2048):
    prefetch.remote()
    calls = {}
    for bench in BENCHMARKS:
        for name, (method, steps, k) in CONFIGS.items():
            if method == "flashmap":
                calls[bench, name] = run_flashmap.spawn(bench, num_prompts, max_new_tokens, steps, k)
            else:
                calls[bench, name] = run_dflash.spawn(bench, num_prompts, max_new_tokens)
    results = {bench: {name: calls[bench, name].get() for name in CONFIGS} for bench in BENCHMARKS}

    with open("resultats_comparaison.json", "w") as f:
        json.dump({"num_prompts": num_prompts, "max_new_tokens": max_new_tokens,
                   "prompt_par_prompt": results}, f, indent=2)
    summarize(results)