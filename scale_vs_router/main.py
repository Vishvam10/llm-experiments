import argparse
import json
from collections import Counter

import requests
from datasets import load_dataset

# ------------------------------------------------------------------------------
# ARGS
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode", type=str, required=True, choices=["big", "routed", "vote"]
)

parser.add_argument("--big-url", type=str, default="http://localhost:8000")
parser.add_argument("--code-url", type=str, default="http://localhost:8001")
parser.add_argument("--math-url", type=str, default="http://localhost:8002")
parser.add_argument("--logic-url", type=str, default="http://localhost:8003")

parser.add_argument("--max-samples", type=int, default=200)

args = parser.parse_args()


# ------------------------------------------------------------------------------
# CALL SERVER
# ------------------------------------------------------------------------------
def call(url, prompt):
    r = requests.post(
        f"{url}/generate",
        json={"prompt": prompt, "max_tokens": 128, "temperature": 0.2},
        timeout=300,
    )
    return r.json()["text"]


# ------------------------------------------------------------------------------
# ROUTER
# ------------------------------------------------------------------------------
def route(prompt):
    if "def " in prompt or "function" in prompt:
        return "code"
    elif any(x in prompt.lower() for x in ["calculate", "+", "-", "*", "/"]):
        return "math"
    else:
        return "logic"


# ------------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------------
def load_data(n):
    gsm = load_dataset("gsm8k", "main", split="test").shuffle().select(range(n))
    humaneval = load_dataset("openai_humaneval", split="test").select(range(50))
    boolq = load_dataset("boolq", split="validation").shuffle().select(range(n))

    data = []

    for x in gsm:
        data.append(
            {
                "task": "math",
                "prompt": x["question"],
                "answer": x["answer"].split("####")[-1].strip(),
            }
        )

    for x in humaneval:
        data.append({"task": "code", "prompt": x["prompt"], "test": x["test"]})

    for x in boolq:
        data.append(
            {
                "task": "logic",
                "prompt": x["question"],
                "answer": "yes" if x["answer"] else "no",
            }
        )

    return data


dataset = load_data(args.max_samples)


# ------------------------------------------------------------------------------
# EVAL
# ------------------------------------------------------------------------------
def eval_math(out, ans):
    return ans in out


def eval_logic(out, ans):
    return ans.lower() in out.lower()


def eval_code(out, test):
    try:
        env = {}
        exec(out, env)
        exec(test, env)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------
# RUN BIG
# ------------------------------------------------------------------------------
def run_big():
    results = []
    for s in dataset:
        out = call(args.big_url, s["prompt"])

        if s["task"] == "math":
            ok = eval_math(out, s["answer"])
        elif s["task"] == "logic":
            ok = eval_logic(out, s["answer"])
        else:
            ok = eval_code(out, s["test"])

        results.append(int(ok))
    return results


# ------------------------------------------------------------------------------
# RUN ROUTED
# ------------------------------------------------------------------------------
def run_routed():
    results = []
    for s in dataset:
        expert = route(s["prompt"])

        url = {
            "code": args.code_url,
            "math": args.math_url,
            "logic": args.logic_url,
        }[expert]

        out = call(url, s["prompt"])

        if s["task"] == "math":
            ok = eval_math(out, s["answer"])
        elif s["task"] == "logic":
            ok = eval_logic(out, s["answer"])
        else:
            ok = eval_code(out, s["test"])

        results.append(int(ok))
    return results


# ------------------------------------------------------------------------------
# RUN VOTE
# ------------------------------------------------------------------------------
def run_vote():
    results = []
    for s in dataset:
        outputs = {
            "code": call(args.code_url, s["prompt"]),
            "math": call(args.math_url, s["prompt"]),
            "logic": call(args.logic_url, s["prompt"]),
        }

        final = Counter(outputs.values()).most_common(1)[0][0]

        if s["task"] == "math":
            ok = eval_math(final, s["answer"])
        elif s["task"] == "logic":
            ok = eval_logic(final, s["answer"])
        else:
            ok = eval_code(final, s["test"])

        results.append(int(ok))
    return results


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if args.mode == "big":
    res = run_big()

elif args.mode == "routed":
    res = run_routed()

else:
    res = run_vote()

acc = sum(res) / len(res)

print(f"\nMODE: {args.mode}")
print(f"Accuracy: {acc:.4f}")

# ------------------------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------------------------
try:
    with open("results.json", "r") as f:
        data = json.load(f)
except Exception:
    data = {}

data[args.mode] = acc

with open("results.json", "w") as f:
    json.dump(data, f, indent=2)

print("\nSaved to results.json")
