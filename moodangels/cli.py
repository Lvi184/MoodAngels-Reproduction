from __future__ import annotations
import argparse, json, dataclasses
from pathlib import Path
from .pipeline import MoodAngelsPipeline
from .data import load_cases


def _asdict(obj):
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj

def cmd_predict(args):
    pipe = MoodAngelsPipeline(args.retrieval_data)
    with open(args.case_json, encoding="utf-8") as f:
        case = json.load(f)
    if isinstance(case, list):
        case = case[0]
    res = pipe.diagnose_dict(case, agent=args.agent)
    print(json.dumps(_asdict(res), ensure_ascii=False, indent=2))

def cmd_evaluate(args):
    pipe = MoodAngelsPipeline(args.retrieval_data or args.data)
    cases = load_cases(args.data)
    n = ok = 0
    rows = []
    for c in cases[:args.limit or len(cases)]:
        if c.label not in (0,1):
            continue
        res = pipe.diagnose_dict(c.raw, agent=args.agent)
        n += 1; ok += int(res.label == c.label)
        rows.append({"id": c.id, "gold": c.label, "pred": res.label, "confidence": res.confidence})
    print(json.dumps({"n": n, "accuracy": ok / n if n else None, "rows": rows[:20]}, ensure_ascii=False, indent=2))

def main(argv=None):
    p = argparse.ArgumentParser("moodangels")
    sub = p.add_subparsers(required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--case-json", required=True)
    pred.add_argument("--retrieval-data", default="data/syn_train.json")
    pred.add_argument("--agent", choices=["raw","display","compare","multi"], default="multi")
    pred.set_defaults(func=cmd_predict)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--data", default="data/syn_train.json")
    ev.add_argument("--retrieval-data", default=None)
    ev.add_argument("--agent", choices=["raw","display","compare","multi"], default="multi")
    ev.add_argument("--limit", type=int, default=100)
    ev.set_defaults(func=cmd_evaluate)
    args = p.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
