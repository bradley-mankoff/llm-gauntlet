"""Run executor benchmarks against DeepSeek v4 Flash."""
import json, time, sys
sys.path.insert(0, '.')
from openai import OpenAI
from tasks import humaneval, ifeval

API_KEY = "sk-29114a6f095f42449e2732b341029b81"
MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com/v1"

def run_benchmark(task, name, thinking_mode, max_tokens, n_samples):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    kwargs = {"n_samples": n_samples, "max_tokens": max_tokens, "seed": 42}
    label = f"{name}_{thinking_mode}"
    print(f"\n=== {label} ===")
    
    t0 = time.time()
    summary, results = task.run(client, MODEL, **kwargs)
    wall = round(time.time() - t0, 1)
    summary["wall_time_sec"] = wall
    summary["thinking_mode"] = thinking_mode
    
    out = f"results/{label}.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    if name == "humaneval":
        print(f"  pass={summary['passed']}/{summary['n_samples']} ({summary['pass_rate']:.1%}) wall={wall}s")
    elif name == "ifeval":
        print(f"  strict={summary['strict_pass_rate']:.1%} loose={summary['loose_pass_rate']:.1%} wall={wall}s")
    return summary
# HumanEval no-thinking (xhigh already done)
run_benchmark(humaneval, "humaneval_ds_flash", "none", 2048, 50)

# IFEval: n=100, max_tokens=8192
for thinking in ["xhigh", "none"]:
    run_benchmark(ifeval, "ifeval_ds_flash", thinking, 8192, 100)
