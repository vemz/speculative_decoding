import json
import modal

app = modal.App("flashmap-repro")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashmap==0.1.0", "transformers>=5.16,<5.21", "torch>=2.9,<2.14")
    .env({"HF_HOME": "/cache"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)  

TARGET, DRAFT = "Qwen/Qwen3-8B", "j-shen/FlashMap-Qwen3-8B"
DATA = "https://raw.githubusercontent.com/two-delta/DCOCO/main/data/{}.jsonl"
BENCHMARKS = ["gsm8k", "humaneval", "mt-bench"]

PAPER = {
    "gsm8k":     {1: 7.69, 2: 8.31, "DFlash": 7.43},
    "humaneval": {1: 5.82, 2: 6.26, "DFlash": 5.74},
    "mt-bench":  {1: 3.34, 2: 3.40, "DFlash": 3.14},
}

@app.function(image=image, gpu="H100", volumes={"/cache": hf_cache}, timeout=3 * 3600)
def run(num_prompts: int, max_new_tokens: int, benchmarks: list) -> dict:
    import urllib.request
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16).cuda().eval()
    draft = AutoModel.from_pretrained(DRAFT, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    hf_cache.commit()

    def encode(prompt):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt", return_dict=True,
        )["input_ids"].cuda()

    results = {}
    for bench in benchmarks:
        lines = urllib.request.urlopen(DATA.format(bench)).read().decode().splitlines()
        prompts = [json.loads(line)["turns"][0] for line in lines[:num_prompts]]  # premier tour seulement
        for steps in (1, 2):
            draft.spec_generate(target, encode(prompts[0]), max_new_tokens=32, num_steps=steps)  # warmup
            committed = rounds = 0
            for prompt in prompts:
                out = draft.spec_generate(target, encode(prompt), max_new_tokens=max_new_tokens,
                                          temperature=0.0, num_steps=steps)
                committed += sum(out.acceptance_lengths)
                rounds += len(out.acceptance_lengths)
            results[f"{bench}/{steps}"] = committed / rounds
            print(f"{bench:<10} {steps} pas : γ = {committed / rounds:.2f}  (papier {PAPER[bench][steps]})", flush=True)
    return results

@app.local_entrypoint()
def main(num_prompts: int = 80, max_new_tokens: int = 2048, benchmarks: str = ",".join(BENCHMARKS)):
    benches = benchmarks.split(",")
    res = run.remote(num_prompts, max_new_tokens, benches)
    print(f"\n{'benchmark':<12}{'pas':>4}{'γ mesuré':>11}{'γ papier':>10}{'DFlash (papier)':>17}")
    for bench in benches:
        for steps in (1, 2):
            print(f"{bench:<12}{steps:>4}{res[f'{bench}/{steps}']:>11.2f}"
                  f"{PAPER[bench][steps]:>10.2f}{PAPER[bench]['DFlash']:>17.2f}")
    with open("resultats_flashmap.json", "w") as f:
        json.dump({"num_prompts": num_prompts, "max_new_tokens": max_new_tokens, "gamma": res}, f, indent=2)