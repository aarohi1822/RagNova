"""Agreement between your manual faithfulness labels and the LLM judge.

Reads eval/judge/labeling_sheet.csv (your yes/no in the `faithful` column) and
eval/judge/judge_outputs.jsonl, binarises judge scores at --threshold, and
reports raw agreement, Cohen's kappa (95% bootstrap CI), the confusion matrix,
a threshold sweep, and every disagreement, for each judge variant.

  python eval/judge_agreement.py [--threshold 0.5]
Writes eval/results/judge_agreement.{json,md}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from eval_metrics import bootstrap_ci, cohen_kappa, confusion  # noqa: E402
from seeds import DEFAULT_SEED  # noqa: E402

SHEET_PATH = EVAL_DIR / "judge" / "labeling_sheet.csv"
JUDGE_PATH = EVAL_DIR / "judge" / "judge_outputs.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
YES, NO = {"yes", "y", "1", "true"}, {"no", "n", "0", "false"}


def load_labels() -> dict[str, bool]:
    with SHEET_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    labels, bad = {}, []
    for row in rows:
        value = row["faithful"].strip().lower()
        if value in YES:
            labels[row["item_id"]] = True
        elif value in NO:
            labels[row["item_id"]] = False
        else:
            bad.append(f"{row['item_id']}={row['faithful']!r}")
    if bad:
        sys.exit(f"{len(bad)} of {len(rows)} rows lack a yes/no label in `faithful`: "
                 + ", ".join(bad))
    return labels


def kappa_stat(pairs: list[tuple[bool, bool]]) -> float:
    k = cohen_kappa([h for h, _ in pairs], [j for _, j in pairs])
    return 0.0 if math.isnan(k) else k


def analyse(labels: dict[str, bool], judged: list[dict], variant: str, threshold: float) -> dict:
    unparsed = [r["item_id"] for r in judged if r["faithfulness"][variant] is None]
    rows = [r for r in judged if r["faithfulness"][variant] is not None]
    human = [labels[r["item_id"]] for r in rows]
    scores = [r["faithfulness"][variant] for r in rows]
    judge = [s >= threshold for s in scores]
    pairs = list(zip(human, judge, strict=True))
    sweep = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        jt = [s >= t for s in scores]
        sweep[t] = {"agreement": sum(h == j for h, j in zip(human, jt, strict=True)) / len(rows),
                    "kappa": cohen_kappa(human, jt)}
    return {
        "n": len(rows),
        "unparseable_judge_outputs": unparsed,
        "human_yes_rate": sum(human) / len(rows),
        "judge_yes_rate": sum(judge) / len(rows),
        "agreement": sum(h == j for h, j in pairs) / len(rows),
        "kappa": cohen_kappa(human, judge),
        "kappa_ci": bootstrap_ci(pairs, stat=kappa_stat, seed=DEFAULT_SEED),
        "confusion": confusion(human, judge),
        "threshold_sweep": sweep,
        "disagreements": [
            {"item_id": r["item_id"], "qid": r["qid"], "human": h, "judge_score": s}
            for r, h, s, j in zip(rows, human, scores, judge, strict=True) if h != j
        ],
    }


def to_markdown(out: dict) -> str:
    lines = ["# LLM-judge reliability", "",
             f"{out['n_labeled']} manually labeled answers; judge backend `{out['backend']}`; "
             f"judge says *faithful* when score ≥ {out['threshold']}.", ""]
    lines += ["| Judge variant | Agreement | Cohen's κ [95% CI] | Human yes-rate | Judge yes-rate "
              "| Unparseable |", "|---|---|---|---|---|---|"]
    for v, r in out["variants"].items():
        lo, hi = r["kappa_ci"]
        lines.append(f"| {v} | {r['agreement']:.2f} | {r['kappa']:.2f} [{lo:.2f}–{hi:.2f}] "
                     f"| {r['human_yes_rate']:.2f} | {r['judge_yes_rate']:.2f} "
                     f"| {len(r['unparseable_judge_outputs'])} |")
    for v, r in out["variants"].items():
        c = r["confusion"]
        lines += ["", f"## {v}", "",
                  "| | judge yes | judge no |", "|---|---|---|",
                  f"| **human yes** | {c['both_yes']} | {c['human_yes_judge_no']} |",
                  f"| **human no** | {c['human_no_judge_yes']} | {c['both_no']} |", "",
                  "Threshold sweep: " + ", ".join(
                      f"{t}: κ={s['kappa']:.2f}" for t, s in r["threshold_sweep"].items()), ""]
        if r["disagreements"]:
            lines.append("Disagreements: " + ", ".join(
                f"{d['item_id']} (human={'yes' if d['human'] else 'no'}, "
                f"judge={d['judge_score']})" for d in r["disagreements"]))
    lines += ["", "κ guide (Landis & Koch): <0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, "
              "0.61–0.80 substantial, >0.80 almost perfect. With n≈30 the CI is wide; "
              "treat the point estimate as indicative."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    labels = load_labels()
    judged = [json.loads(line) for line in JUDGE_PATH.open(encoding="utf-8")]
    missing = [r["item_id"] for r in judged if r["item_id"] not in labels]
    if missing:
        sys.exit(f"judge items missing from the labeling sheet: {missing}")

    variants = list(judged[0]["faithfulness"])
    out = {
        "n_labeled": len(labels), "threshold": args.threshold, "backend": judged[0]["backend"],
        "variants": {v: analyse(labels, judged, v, args.threshold) for v in variants},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "judge_agreement.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    md = to_markdown(out)
    (RESULTS_DIR / "judge_agreement.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
