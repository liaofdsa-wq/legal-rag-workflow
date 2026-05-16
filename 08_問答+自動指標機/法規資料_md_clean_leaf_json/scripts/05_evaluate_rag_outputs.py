from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator import DEFAULT_BAD_CASE_THRESHOLD, DEFAULT_EMBEDDING_MODEL, evaluate_file, export_evaluation_result


EVAL_ROOT = ROOT / "data" / "evaluation"


def default_output_dir(input_path: Path) -> Path:
    return EVAL_ROOT / input_path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--bad-case-threshold", type=float, default=DEFAULT_BAD_CASE_THRESHOLD)
    args = parser.parse_args()

    result = evaluate_file(
        input_path=args.input,
        model_name=args.model,
        bad_case_threshold=args.bad_case_threshold,
    )
    output_dir = args.output_dir or default_output_dir(args.input)
    outputs = export_evaluation_result(result, output_dir)

    print(
        json.dumps(
            {
                "summary": result["summary"],
                "outputs": {key: str(value) for key, value in outputs.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
