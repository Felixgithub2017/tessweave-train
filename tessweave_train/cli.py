import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .core import advise, build_plan
from .runtime import doctor, run


def choose_goal():
    print("What supervision do you have?\n1. Correct demonstrations → SFT\n2. Chosen/rejected answers → DPO\n3. Verifiable math answers → GRPO\n4. Compatible scalar reward checkpoint → PPO\n5. Preference pairs to train a reward model → RM")
    goal = {"1": "examples", "2": "preferences", "3": "math", "4": "reward-model", "5": "train-reward"}.get(input("Choose 1–5: ").strip())
    if goal is None:
        raise ValueError("Choose a listed supervision type")
    return goal


def wizard():
    route = advise(choose_goal())
    print("\nRecommended first experiment: " + route["method"].upper() + " + LoRA")
    print(route["why"])
    print("Schema examples live in examples/. Supply your own reviewed data for useful training; bundled examples only test the pipeline.")
    def ask(label, default=None):
        value = input(label + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not value and default is None:
            raise ValueError(label + " is required; prepare that input before planning")
        return value or default
    spec = {"method": route["method"], "tuner": "lora",
            "model": ask("Complete local checkpoint directory"),
            "train": ask("Training JSONL path"), "eval": ask("Independent evaluation JSONL path"),
            "python": ask("Trusted backend Python with ms-swift 4.5.3", sys.executable),
            "gpus": int(ask("GPUs on this Linux host (no automatic rental)", "1")),
            "max_length": int(ask("Token budget including completion for online RL", "2048")),
            "output": ask("New experiment directory", "runs/first-smoke")}
    if route["method"] == "ppo":
        spec["reward_model"] = ask("Compatible complete scalar reward checkpoint")
    print("First run is 20 optimizer steps (PPO: rollout/update cycles), BF16, microbatch 1, LoRA rank 8; this is a smoke test, not a quality-training schedule.")
    return build_plan(spec)


def save_plan(plan, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print("Command preview (NOT executed):\n" + shlex.join(plan["argv"]))
    print("Approval SHA256: " + plan["approval_sha256"])
    print("Next: inspect the plan, run doctor, then run --approve SHA256. GPU validation is still required.")


def main():
    parser = argparse.ArgumentParser(description="TessWeave Train: prepare → review → execute. No automatic installs or weight downloads.")
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("guide", help="Choose an algorithm from your available supervision")
    p.add_argument("--goal", choices=["examples", "preferences", "math", "reward-model", "train-reward"])
    p = subs.add_parser("wizard", help="Answer guided questions and generate a plan without editing a spec file")
    p.add_argument("--out", required=True)
    p = subs.add_parser("plan", help="Validate local inputs and produce a reviewable command, without running training")
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    for name in ("doctor", "run"):
        p = subs.add_parser(name)
        p.add_argument("--plan", required=True)
        if name == "run": p.add_argument("--approve", required=True, help="Complete SHA256 shown in the reviewed plan")
    args = parser.parse_args()
    try:
        if args.command == "guide":
            goal = args.goal or choose_goal()
            print(json.dumps(advise(goal), indent=2, ensure_ascii=False))
            return
        if args.command == "wizard":
            save_plan(wizard(), args.out)
            return
        if args.command == "plan":
            plan = build_plan(json.loads(Path(args.spec).read_text()))
            save_plan(plan, args.out)
            return
        plan = json.loads(Path(args.plan).read_text())
        if args.command == "doctor":
            report = doctor(plan)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            raise SystemExit(2 if report["blockers"] else 0)
        raise SystemExit(run(plan, args.approve))
    except (ValueError, OSError, KeyError, EOFError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"Action required: {exc}\n")


if __name__ == "__main__":
    main()
