"""Agreement between manual faithfulness labels and the LLM judge.

Raters:
  human   - eval/judge/labeling_sheet.csv, `faithful` column (yes / no / unsure)
  claude  - eval/judge/claude_labels.csv (optional second annotator, labeled
            independently before the human labels were collected)

For each judge variant the judge score is binarised at --threshold. Items a rater
marked `unsure` are excluded from that rater's comparisons and counted separately.
Reports raw agreement, Cohen's kappa (95% bootstrap CI), confusion matrix, a
threshold sweep and every disagreement; plus human-vs-claude inter-annotator
agreement when both label files exist.

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
CLAUDE_PATH = EVAL_DIR / "judge" / "claude_labels.csv"
JUDGE_PATH = EVAL_DIR / "judge" / "judge_outputs.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
YES = {"yes", "y", "1", "true"}
NO = {"no", "n", "0", "false"}
UNSURE = {"unsure", "idk", "i don't know", "i dont know", "dont know", "?"}
SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_labels(path: Path) -> dict[str, bool | None]:
    """item_id -> True (faithful) / False / None (unsure). Exits if any row is blank."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    labels: dict[str, bool | None] = {}
    bad = []
    for row in rows:
        value = row["faithful"].strip().lower()
        if value in YES:
            labels[row["item_id"]] = True
        elif value in NO:
            labels[row["item_id"]] = False
        elif value in UNSURE:
            labels[row["item_id"]] = None
        else:
            bad.append(f"{row['item_id']}={row['faithful']!r}")
    if bad:
        sys.exit(f"{path.name}: {len(bad)} of {len(rows)} rows lack a yes/no/unsure label: "
                 + ", ".join(bad))
    return labels


def kappa_stat(pairs: list[tuple[bool, bool]]) -> float:
    k = cohen_kappa([a for a, _ in pairs], [b for _, b in pairs])
    return 0.0 if math.isnan(k) else k


def compare(a: list[bool], b: list[bool]) -> dict:
    pairs = list(zip(a, b, strict=True))
    return {
        "n": len(pairs),
        "agreement": sum(x == y for x, y in pairs) / len(pairs),
        "kappa": cohen_kappa(a, b),
        "kappa_ci": bootstrap_ci(pairs, stat=kappa_stat, seed=DEFAULT_SEED),
        "confusion": confusion(a, b),
    }


def rater_vs_judge(labels: dict, judged: list[dict], variant: str, threshold: float) -> dict:
    unparsed = [r["item_id"] for r in judged if r["faithfulness"][variant] is None]
    unsure = [r["item_id"] for r in judged if labels[r["item_id"]] is None]
    rows = [r for r in judged
            if r["faithfulness"][variant] is not None and labels[r["item_id"]] is not None]
    rater = [labels[r["item_id"]] for r in rows]
    scores = [r["faithfulness"][variant] for r in rows]
    judge = [s >= threshold for s in scores]
    out = compare(rater, judge)
    out.update({
        "excluded_unsure": unsure,
        "unparseable_judge_outputs": unparsed,
        "rater_yes_rate": sum(rater) / len(rater),
        "judge_yes_rate": sum(judge) / len(judge),
        "threshold_sweep": {
            t: {"agreement": sum(h == (s >= t) for h, s in zip(rater, scores, strict=True))
                / len(rater),
                "kappa": cohen_kappa(rater, [s >= t for s in scores])}
            for t in SWEEP
        },
        "disagreements": [
            {"item_id": r["item_id"], "qid": r["qid"], "rater": h, "judge_score": s}
            for r, h, s, j in zip(rows, rater, scores, judge, strict=True) if h != j
        ],
    })
    return out


def rater_vs_rater(a: dict, b: dict) -> dict:
    items = sorted(k for k in a if a[k] is not None and b.get(k) is not None)
    out = compare([a[k] for k in items], [b[k] for k in items])
    out["disagreements"] = [{"item_id": k, "human": a[k], "claude": b[k]}
                            for k in items if a[k] != b[k]]
    out["excluded_unsure"] = sorted(k for k in a if a[k] is None or b.get(k) is None)
    return out


def _fmt_kappa(r: dict) -> str:
    lo, hi = r["kappa_ci"]
    k = "n/a" if math.isnan(r["kappa"]) else f"{r['kappa']:.2f}"
    return f"{k} [{lo:.2f}–{hi:.2f}]"


def to_markdown(out: dict) -> str:
    lines = ["# LLM-judge reliability", "",
             f"{out['n_items']} answers; judge `{out['backend']}` says *faithful* when "
             f"score ≥ {out['threshold']}. Items a rater marked *unsure* are excluded from "
             "that rater's comparisons.", "",
             "| Comparison | n | Agreement | Cohen's κ [95% CI] | Rater yes-rate "
             "| Judge yes-rate |", "|---|---|---|---|---|---|"]
    for rater, by_variant in out["vs_judge"].items():
        for variant, r in by_variant.items():
            lines.append(f"| {rater} vs judge ({variant}) | {r['n']} | {r['agreement']:.2f} "
                         f"| {_fmt_kappa(r)} | {r['rater_yes_rate']:.2f} "
                         f"| {r['judge_yes_rate']:.2f} |")
    if out.get("human_vs_claude"):
        r = out["human_vs_claude"]
        lines.append(f"| human vs claude (inter-annotator) | {r['n']} | {r['agreement']:.2f} "
                     f"| {_fmt_kappa(r)} | – | – |")
    for rater, by_variant in out["vs_judge"].items():
        for variant, r in by_variant.items():
            c = r["confusion"]
            sweep = ", ".join(
                f"{t}: κ={'n/a' if math.isnan(s['kappa']) else format(s['kappa'], '.2f')}"
                for t, s in r["threshold_sweep"].items())
            lines += ["", f"## {rater} vs judge ({variant})", "",
                      f"Excluded as unsure: {', '.join(r['excluded_unsure']) or 'none'}", "",
                      "| | judge yes | judge no |", "|---|---|---|",
                      f"| **{rater} yes** | {c['both_yes']} | {c['human_yes_judge_no']} |",
                      f"| **{rater} no** | {c['human_no_judge_yes']} | {c['both_no']} |", "",
                      f"Threshold sweep: {sweep}", ""]
            if r["disagreements"]:
                lines.append("Disagreements: " + ", ".join(
                    f"{d['item_id']} ({rater}={'yes' if d['rater'] else 'no'}, "
                    f"judge={d['judge_score']})" for d in r["disagreements"]))
    if out.get("human_vs_claude"):
        r = out["human_vs_claude"]
        lines += ["", "## human vs claude", "",
                  "Disagreements: " + (", ".join(
                      f"{d['item_id']} (human={'yes' if d['human'] else 'no'}, "
                      f"claude={'yes' if d['claude'] else 'no'})"
                      for d in r["disagreements"]) or "none")]
    lines += ["", "κ guide (Landis & Koch): <0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, "
              "0.61–0.80 substantial, >0.80 almost perfect. κ is 0 by construction when a "
              "rater gives the same label to every item. With n≤30 the CI is wide."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    judged = [json.loads(line) for line in JUDGE_PATH.open(encoding="utf-8")]
    raters = {"human": load_labels(SHEET_PATH)}
    if CLAUDE_PATH.exists():
        raters["claude"] = load_labels(CLAUDE_PATH)
    for name, labels in raters.items():
        missing = [r["item_id"] for r in judged if r["item_id"] not in labels]
        if missing:
            sys.exit(f"{name}: judge items missing from labels: {missing}")

    variants = list(judged[0]["faithfulness"])
    out = {
        "n_items": len(judged), "threshold": args.threshold, "backend": judged[0]["backend"],
        "unsure": {n: sorted(k for k, v in lab.items() if v is None) for n, lab in raters.items()},
        "vs_judge": {name: {v: rater_vs_judge(lab, judged, v, args.threshold) for v in variants}
                     for name, lab in raters.items()},
    }
    out["variants"] = out["vs_judge"]["human"]  # human-vs-judge, the headline comparison
    if "claude" in raters:
        out["human_vs_claude"] = rater_vs_rater(raters["human"], raters["claude"])

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "judge_agreement.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n")
    md = to_markdown(out)
    (RESULTS_DIR / "judge_agreement.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
