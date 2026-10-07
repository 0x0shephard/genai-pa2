import argparse
import json


def mappings(values):
    return dict(value.split("=", 1) for value in values)


def main():
    parser = argparse.ArgumentParser(description="PA2 alignment pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--track", choices=["sft", "hh", "ultrafeedback", "persona", "alpaca"], required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--train-size", type=int)
    prepare.add_argument("--val-size", type=int, default=1000)
    prepare.add_argument("--seed", type=int, default=42)
    for kind in ("sft", "dpo"):
        train = commands.add_parser(kind)
        train.add_argument("--data", required=True)
        train.add_argument("--output", required=True)
        train.add_argument("--model")
        train.add_argument("--resume")
        for field in ("max-steps", "batch-size", "accumulation", "eval-steps", "epochs", "max-length", "max-prompt-length", "seed"):
            train.add_argument("--" + field, type=int)
        for field in ("learning-rate", "beta", "weight-decay", "warmup-ratio"):
            train.add_argument("--" + field, type=float)
        train.add_argument("--scheduler", choices=["constant", "cosine"])
        train.add_argument("--policy-device")
        train.add_argument("--reference-device")
    persona = commands.add_parser("persona")
    persona.add_argument("--data", required=True)
    persona.add_argument("--output", required=True)
    persona.add_argument("--max-steps", type=int)
    persona.add_argument("--resume")
    generation = commands.add_parser("generate")
    generation.add_argument("--checkpoint", required=True)
    generation.add_argument("--prompts", required=True)
    generation.add_argument("--output", required=True)
    generation.add_argument("--generator", required=True)
    generation.add_argument("--persona", action="store_true")
    generation.add_argument("--limit", type=int)
    generation.add_argument("--resume-from")
    generation.add_argument("--max-new-tokens", type=int, default=256)
    generation.add_argument("--repetition-penalty", type=float, default=1.)
    generation.add_argument("--no-repeat-ngram-size", type=int, default=0)
    generation.add_argument("--device", default="cuda:0")
    pairs = commands.add_parser("pairs")
    pairs.add_argument("--baseline", required=True)
    pairs.add_argument("--candidate", required=True)
    pairs.add_argument("--output", required=True)
    pairs.add_argument("--track", choices=["hh", "ultrafeedback", "persona"], required=True)
    judge = commands.add_parser("judge")
    judge.add_argument("--pairs", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--judge-model", default="google/gemini-3.6-flash")
    judge.add_argument("--batch-size", type=int, default=5)
    judge.add_argument("--sleep-seconds", type=float, default=5)
    judge.add_argument("--max-retries", type=int, default=3)
    judge.add_argument("--resume-from")
    plot = commands.add_parser("plot")
    plot.add_argument("--run", required=True)
    plot.add_argument("--output", required=True)
    plot.add_argument("--name")
    report = commands.add_parser("report")
    report.add_argument("--run", action="append", default=[], help="name=/path/to/run")
    report.add_argument("--judge", action="append", default=[], help="track=/path/to/judgments")
    report.add_argument("--comparison", action="append", default=[], help="track=/path/to/pairs.jsonl")
    report.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        from .data import prepare_alpaca, prepare_persona, prepare_track
        track = args.pop("track")
        if track == "alpaca":
            result = prepare_alpaca(args["output"], args["train_size"] or 300, args["seed"])
        elif track == "persona":
            result = prepare_persona(args["output"], args["seed"])
        else:
            result = prepare_track(track, **args)
    elif command in ("sft", "dpo"):
        from .train import run_training
        args["data_dir"] = args.pop("data")
        result = run_training(command, **args)
    elif command == "persona":
        from .persona import run_persona
        args["data_dir"] = args.pop("data")
        result = run_persona(**args)
    elif command == "generate":
        from .generate import generate_outputs
        args["prompts_path"] = args.pop("prompts")
        args["output_path"] = args.pop("output")
        result = generate_outputs(**args)
    elif command == "pairs":
        from .generate import build_pairs
        args["first_path"] = args.pop("baseline")
        args["second_path"] = args.pop("candidate")
        result = build_pairs(**args)
    elif command == "judge":
        from .judge import judge_pairs
        args["pairs_path"] = args.pop("pairs")
        args["output_dir"] = args.pop("output")
        result = judge_pairs(**args)
    elif command == "plot":
        from .report import plot_run
        args["run_dir"] = args.pop("run")
        result = plot_run(**args)
    else:
        from .report import build_report
        args["run_dirs"] = mappings(args.pop("run"))
        args["judge_dirs"] = mappings(args.pop("judge"))
        args["comparisons"] = mappings(args.pop("comparison"))
        result = build_report(**args)
    print(json.dumps(result, indent=2) if isinstance(result, dict) else result)


if __name__ == "__main__":
    main()
