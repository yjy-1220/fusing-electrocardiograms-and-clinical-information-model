
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow direct execution via `python model/text_process/clinical_text_processing.py`:
# insert the repo root into sys.path so the `model` package can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.text_process.config import TextProcessingConfig
from model.text_process.pipeline import ClinicalTextProcessor


def _print_stats(title: str, stats) -> None:
    print(f"{title}:")
    print(
        f"  n={stats['count']} | mean={stats['mean']} | "
        f"median={stats['median']} | p95={stats['p95']} | "
        f"max={stats['max']} | over_limit_ratio={stats['over_ratio'] * 100:.1f}%"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Chinese output on Windows console

    parser = argparse.ArgumentParser(
        description="Clinical text pipeline (extraction/QC + length normalization)"
    )
    parser.add_argument("--input", required=True,
                        help="EMR export file (json / jsonl / csv)")
    parser.add_argument("--model", type=str, default="hfl/chinese-macbert-base",
                        help="HuggingFace model name (default: MacBERT)")
    parser.add_argument("--offline", action="store_true",
                        help="Skip model download; use the char-level fallback tokenizer")
    parser.add_argument("--max-tokens", type=int, default=512, help="Total input length limit")
    parser.add_argument("--head-ratio", type=float, default=0.6, help="Head budget ratio")
    parser.add_argument("--no-tags", action="store_true", help="Do not add type tags when joining")
    args = parser.parse_args()

    cfg = TextProcessingConfig(
        model_name=args.model,
        max_tokens=args.max_tokens,
        head_ratio=args.head_ratio,
        offline=args.offline,
        tag_segments=not args.no_tags,
    )
    proc = ClinicalTextProcessor(cfg)
    result = proc.run(args.input)

    print(f"\n===== Stage 1: extraction & QC (tokenizer backend: {proc.tokenizer.backend}) =====")
    print(f"Input docs: {result['input_docs']} | kept: {result['kept_docs']} | "
          f"dropped: {result['dropped_docs']} | patients: {result['patients']}")

    print("\n===== Stage 2a: length distribution (chars; 1 Chinese char ~ 1 token) =====")
    _print_stats("Char length", result["char_stats"])
    print("\n===== Stage 2a: length distribution (tokens) =====")
    _print_stats("Token length", result["token_stats"])

    print("\n===== Stage 2b: head-tail truncation results =====")
    for pid, info in result["truncated"].items():
        if info["truncated"]:
            sample = proc.tokenizer.decode(info["input_ids"])
            print(f"Patient {pid}: {info['raw_len']} tokens -> {info['final_len']} tokens "
                  f"(head {info['head_kept']} + tail {info['tail_kept']})")
            print(f"  Kept head: {sample[:80]}…")
            print(f"  Kept tail: …{sample[-80:]}")
        else:
            print(f"Patient {pid}: {info['raw_len']} tokens, within limit, kept intact")
    print("\nDone.")


if __name__ == "__main__":
    main()
