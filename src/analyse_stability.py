"""Analyse a completed SHAP sampling grid (RQ1 and the stability side of RQ3).

Reads the output folder written by ``shap_experiment.py`` and writes every
table and figure used in the report into one analysis folder, so all
reported numbers can be regenerated from the raw grid output.

Usage
-----
    python src/analyse_stability.py \\
        --grid artifacts/shap_week10_grid \\
        --output artifacts/analysis_week10

Inputs (from the grid folder)
-----------------------------
    within_condition_stability.csv   one row per configuration
    between_condition_agreement.csv  one row per configuration (marks the reference)
    consensus_rankings.csv           features x configuration, mean |SHAP| over seeds
    rankings_long.csv                one row per run x feature
    runs.csv                         one row per run (runtime, additivity)

Outputs
-------
    Tables (CSV):
      factor_effects_within.csv        mean stability by each design factor
      factor_effects_between.csv       mean agreement with the reference by factor
      kruskal_wallis.csv               H statistic and p-value per factor x metric
      dunn_posthoc.csv                 pairwise Dunn tests, Holm-adjusted
      size_tradeoff.csv                stability and runtime per background x evaluation size
      size_paired_tests.csv            each size vs the largest, matched on all other factors
      size_regression.csv              metric ~ log2(background size)
      top5_boundary.csv                features ranked 4-6 per configuration, with gaps
      feature_importance_by_method.csv median importance and rank per evaluation method
      topk_changes_vs_reference.csv    features entering / leaving the top 10
      top5_swaps_by_run.csv            per run: top-5 features that differ from the reference
      top5_swap_summary.csv            swap rates and the features involved, per evaluation method
      bootstrap_ci.csv                 95% bootstrap CI on each feature's mean |SHAP|, per configuration
    Figures (PNG):
      stability_vs_background_size.png
      top5_by_evaluation_method.png
      temperature_importance_by_method.png
      cost_vs_stability.png
    summary.json                       headline numbers

Notes on the statistics
-----------------------
* Tests use configuration-level values (one row per configuration), not
  individual seed pairs, since pairs within a configuration are not
  independent.
* Kruskal-Wallis tests one factor at a time, pooling over the others. The
  grid is fully crossed, so each level of a factor sees every combination
  of the remaining factors equally often.
* Size comparisons pair each configuration with the configuration that is
  identical except for the size being compared (Wilcoxon signed-rank).
  A non-significant result is NOT evidence of equivalence; read the median
  difference alongside the p-value.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FACTORS = ["background_size", "evaluation_size", "background_method", "evaluation_method"]
WITHIN_METRICS = ["kendalls_w", "spearman_mean", "kendall_mean",
                  "jaccard_top5_mean", "jaccard_top10_mean", "jaccard_top5_min"]
BETWEEN_METRICS = ["spearman_vs_reference", "kendall_vs_reference",
                   "jaccard_top5_vs_reference", "jaccard_top10_vs_reference"]
TEST_METRICS = ["kendalls_w", "jaccard_top5_mean", "jaccard_top10_mean"]
NEAR_TIE_MW = 5.0

COLOURS = ["#1C7293", "#F2A541", "#0A2540", "#B5443C", "#5B8C5A"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_grid(grid_dir: Path) -> dict:
    grid_dir = Path(grid_dir)
    needed = ["within_condition_stability.csv", "between_condition_agreement.csv",
              "consensus_rankings.csv", "rankings_long.csv", "runs.csv"]
    missing = [n for n in needed if not (grid_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"{grid_dir} is missing {missing}. Run shap_experiment.py first.")
    between = pd.read_csv(grid_dir / "between_condition_agreement.csv")
    reference = between.loc[between["is_reference"].astype(bool), "config_id"]
    if len(reference) != 1:
        raise ValueError("between_condition_agreement.csv must mark exactly one reference configuration.")
    return {
        "within": pd.read_csv(grid_dir / "within_condition_stability.csv"),
        "between": between,
        "consensus": pd.read_csv(grid_dir / "consensus_rankings.csv", index_col="feature"),
        "long": pd.read_csv(grid_dir / "rankings_long.csv"),
        "runs": pd.read_csv(grid_dir / "runs.csv"),
        "reference": reference.iloc[0],
    }


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------

def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, returned in the original order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (m - position) * p[index]))
        adjusted[index] = running
    return adjusted


def dunn_test(values: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Dunn's pairwise test after Kruskal-Wallis, with tie correction and Holm adjustment."""
    ranks = stats.rankdata(values.to_numpy())
    data = pd.DataFrame({"rank": ranks, "group": groups.to_numpy()})
    n_total = len(data)
    _, tie_counts = np.unique(ranks, return_counts=True)
    tie_term = float(((tie_counts ** 3) - tie_counts).sum()) / (12 * (n_total - 1))
    base_variance = n_total * (n_total + 1) / 12 - tie_term
    summary = data.groupby("group")["rank"].agg(["mean", "size"])
    rows = []
    for a, b in itertools.combinations(summary.index, 2):
        se = np.sqrt(base_variance * (1 / summary.loc[a, "size"] + 1 / summary.loc[b, "size"]))
        z = (summary.loc[a, "mean"] - summary.loc[b, "mean"]) / se if se > 0 else 0.0
        rows.append({"group_a": a, "group_b": b, "z": z, "p": 2 * stats.norm.sf(abs(z))})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = holm_adjust(result["p"])
    return result


def kruskal_and_dunn(within: pd.DataFrame, metrics=TEST_METRICS):
    kw_rows, dunn_frames = [], []
    for factor, metric in itertools.product(FACTORS, metrics):
        groups = [g[metric].to_numpy() for _, g in within.groupby(factor)]
        if len(groups) < 2 or all(np.ptp(g) == 0 for g in groups) and len({g[0] for g in groups}) == 1:
            continue
        h, p = stats.kruskal(*groups)
        kw_rows.append({"factor": factor, "metric": metric, "levels": len(groups),
                        "configs_per_level": int(np.median([len(g) for g in groups])),
                        "H": h, "p": p})
        dunn = dunn_test(within[metric], within[factor].astype(str))
        dunn.insert(0, "metric", metric)
        dunn.insert(0, "factor", factor)
        dunn_frames.append(dunn)
    kw = pd.DataFrame(kw_rows)
    if not kw.empty:
        kw["p_holm"] = holm_adjust(kw["p"])
    dunn_all = pd.concat(dunn_frames, ignore_index=True) if dunn_frames else pd.DataFrame()
    return kw, dunn_all


# --------------------------------------------------------------------------
# RQ1 tables
# --------------------------------------------------------------------------

def factor_effects(frame: pd.DataFrame, metrics) -> pd.DataFrame:
    parts = []
    for factor in FACTORS:
        table = frame.groupby(factor)[metrics].mean().reset_index().rename(columns={factor: "level"})
        table.insert(0, "factor", factor)
        table["level"] = table["level"].astype(str)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def top5_boundary(consensus: pd.DataFrame, within: pd.DataFrame) -> pd.DataFrame:
    meta = within.set_index("config_id")
    rows = []
    for cid in consensus.columns:
        ordered = consensus[cid].sort_values(ascending=False)
        rows.append({
            "config_id": cid,
            **{c: meta.loc[cid, c] for c in FACTORS},
            "rank4": ordered.index[3], "rank5": ordered.index[4], "rank6": ordered.index[5],
            "gap_4_5": ordered.iloc[3] - ordered.iloc[4],
            "gap_5_6": ordered.iloc[4] - ordered.iloc[5],
            "near_tie_5_6": (ordered.iloc[4] - ordered.iloc[5]) < NEAR_TIE_MW,
            "jaccard_top5_mean": meta.loc[cid, "jaccard_top5_mean"],
        })
    return pd.DataFrame(rows).sort_values("gap_5_6").reset_index(drop=True)


def feature_importance_by_method(consensus: pd.DataFrame, within: pd.DataFrame) -> pd.DataFrame:
    methods = within.set_index("config_id").loc[consensus.columns, "evaluation_method"]
    ranks = consensus.rank(ascending=False, method="min")
    rows = []
    for method in sorted(methods.unique()):
        cols = methods[methods == method].index
        for feature in consensus.index:
            rows.append({"evaluation_method": method, "feature": feature,
                         "median_mean_abs_shap": consensus.loc[feature, cols].median(),
                         "median_rank": ranks.loc[feature, cols].median()})
    return pd.DataFrame(rows).sort_values(["evaluation_method", "median_rank"]).reset_index(drop=True)


def topk_changes_vs_reference(consensus: pd.DataFrame, within: pd.DataFrame,
                              reference: str, k: int = 10) -> pd.DataFrame:
    reference_top = set(consensus[reference].nlargest(k).index)
    meta = within.set_index("config_id")
    rows = []
    for cid in consensus.columns:
        top = set(consensus[cid].nlargest(k).index)
        rows.append({"config_id": cid, **{c: meta.loc[cid, c] for c in FACTORS},
                     "entered": ";".join(sorted(top - reference_top)),
                     "left": ";".join(sorted(reference_top - top)),
                     "n_changed": len(top - reference_top)})
    return pd.DataFrame(rows)


def top5_swaps(long: pd.DataFrame, consensus: pd.DataFrame, reference: str):
    """Per run: which top-5 features differ from the reference top 5."""
    reference_top5 = set(consensus[reference].nlargest(5).index)
    top5 = (long[long["rank"] <= 5]
            .groupby(["run_id", "config_id"] + FACTORS)["feature"]
            .apply(lambda s: set(s)).reset_index())
    top5["entered"] = top5["feature"].apply(lambda s: ";".join(sorted(s - reference_top5)))
    top5["left"] = top5["feature"].apply(lambda s: ";".join(sorted(reference_top5 - s)))
    top5["differs"] = top5["entered"] != ""
    by_run = top5.drop(columns="feature")

    rows = []
    for method, group in by_run.groupby("evaluation_method"):
        entered = pd.Series([f for x in group["entered"] if x for f in x.split(";")]).value_counts()
        left = pd.Series([f for x in group["left"] if x for f in x.split(";")]).value_counts()
        rows.append({"evaluation_method": method, "runs": len(group),
                     "runs_differing": int(group["differs"].sum()),
                     "share_differing": group["differs"].mean(),
                     "entered": "; ".join(f"{k} ({v})" for k, v in entered.items()),
                     "left": "; ".join(f"{k} ({v})" for k, v in left.items())})
    return by_run, pd.DataFrame(rows)


def bootstrap_ci(long: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """95% percentile bootstrap CI on each feature's mean |SHAP|, resampling seeds within a configuration."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid, group in long.groupby("config_id"):
        matrix = group.pivot(index="seed", columns="feature", values="mean_abs_shap")
        values = matrix.to_numpy()
        n = values.shape[0]
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = values[idx].mean(axis=1)  # n_boot x features
        low, high = np.percentile(boot_means, [2.5, 97.5], axis=0)
        means = values.mean(axis=0)
        for j, feature in enumerate(matrix.columns):
            rows.append({"config_id": cid, "feature": feature, "n_seeds": n,
                         "mean_abs_shap": means[j], "ci_low": low[j], "ci_high": high[j]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# RQ3 tables
# --------------------------------------------------------------------------

def size_tradeoff(within: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    stability = within.groupby(["background_size", "evaluation_size"])[
        ["kendalls_w", "kendall_mean", "jaccard_top5_mean", "jaccard_top10_mean"]].mean()
    timing = runs.groupby(["background_size", "evaluation_size"])["elapsed_seconds"].mean().rename(
        "mean_seconds_per_run")
    table = stability.join(timing).reset_index()
    largest = table.loc[(table["background_size"] == table["background_size"].max())
                        & (table["evaluation_size"] == table["evaluation_size"].max())].iloc[0]
    table["cost_vs_largest"] = table["mean_seconds_per_run"] / largest["mean_seconds_per_run"]
    # share of the stability gained between the smallest and largest background size,
    # within the same evaluation size
    for metric in ["kendalls_w", "jaccard_top5_mean"]:
        shares = []
        for _, row in table.iterrows():
            same_eval = table[table["evaluation_size"] == row["evaluation_size"]]
            lo = same_eval.loc[same_eval["background_size"].idxmin(), metric]
            hi = same_eval.loc[same_eval["background_size"].idxmax(), metric]
            shares.append((row[metric] - lo) / (hi - lo) if hi != lo else np.nan)
        table[f"{metric}_share_of_gain"] = shares
    return table


def size_paired_tests(within: pd.DataFrame, metrics=TEST_METRICS) -> pd.DataFrame:
    """Each size vs the largest size, pairing configurations identical in every other factor."""
    rows = []
    for size_col, other in [("background_size", ["background_method", "evaluation_method", "evaluation_size"]),
                            ("evaluation_size", ["background_method", "evaluation_method", "background_size"])]:
        largest = within[size_col].max()
        reference = within[within[size_col] == largest].set_index(other)
        for size in sorted(s for s in within[size_col].unique() if s != largest):
            candidate = within[within[size_col] == size].set_index(other)
            paired = candidate.join(reference, lsuffix="_small", rsuffix="_large", how="inner")
            for metric in metrics:
                diff = paired[f"{metric}_small"] - paired[f"{metric}_large"]
                if np.allclose(diff, 0):
                    p = 1.0
                else:
                    p = stats.wilcoxon(paired[f"{metric}_small"], paired[f"{metric}_large"]).pvalue
                rows.append({"size_factor": size_col, "size": size, "compared_with": largest,
                             "metric": metric, "pairs": len(paired),
                             "median_difference": float(diff.median()),
                             "mean_difference": float(diff.mean()),
                             "share_smaller_is_lower": float((diff < 0).mean()),
                             "p": p})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm_adjust(table["p"])
    return table


def size_regression(within: pd.DataFrame, metrics=TEST_METRICS) -> pd.DataFrame:
    """metric ~ a + b * log2(background_size), overall and per evaluation method."""
    rows = []
    subsets = [("all", within)] + [(m, g) for m, g in within.groupby("evaluation_method")]
    for label, frame in subsets:
        x = np.log2(frame["background_size"].to_numpy(dtype=float))
        for metric in metrics:
            y = frame[metric].to_numpy(dtype=float)
            fit = stats.linregress(x, y)
            rows.append({"subset": label, "metric": metric, "slope_per_doubling": fit.slope,
                         "intercept": fit.intercept, "r_squared": fit.rvalue ** 2, "p": fit.pvalue,
                         "n_configs": len(frame)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stability_vs_size(within: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in [(axes[0], "kendalls_w", "Kendall's W"),
                              (axes[1], "jaccard_top5_mean", "Top-5 Jaccard (mean)")]:
        for i, (ev_size, group) in enumerate(within.groupby("evaluation_size")):
            series = group.groupby("background_size")[metric].mean()
            ax.plot(series.index, series.values, marker="o", color=COLOURS[i],
                    label=f"evaluation size {ev_size}")
        ax.set_xscale("log", base=2)
        sizes = sorted(within["background_size"].unique())
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("Background size")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Within-condition stability by sample size (mean over configurations)")
    _save(fig, path)


def plot_top5_by_method(within: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (method, group) in enumerate(within.groupby("evaluation_method")):
        series = group.groupby("background_size")["jaccard_top5_mean"].mean()
        ax.plot(series.index, series.values, marker="o", color=COLOURS[i % len(COLOURS)], label=method)
    ax.set_xscale("log", base=2)
    sizes = sorted(within["background_size"].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Background size")
    ax.set_ylabel("Top-5 Jaccard (mean over seed pairs)")
    ax.set_title("Top-5 stability by evaluation method")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_temperature_importance(consensus: pd.DataFrame, within: pd.DataFrame, path: Path,
                                features=("temp_max", "temp_min", "temperature")) -> None:
    methods = within.set_index("config_id").loc[consensus.columns, "evaluation_method"]
    order = sorted(methods.unique())
    features = [f for f in features if f in consensus.index]
    width = 0.8 / max(1, len(features))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(order))
    for i, feature in enumerate(features):
        means = [consensus.loc[feature, methods[methods == m].index].mean() for m in order]
        sds = [consensus.loc[feature, methods[methods == m].index].std() for m in order]
        ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=sds, capsize=3,
               color=COLOURS[i % len(COLOURS)], label=feature)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_stratified", "") for m in order])
    ax.set_ylabel("Mean |SHAP| (MW), averaged over configurations")
    ax.set_title("Temperature feature importance by evaluation method (± SD across configurations)")
    ax.grid(axis="y", alpha=0.3)
    if features:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No temperature features found in this grid",
                transform=ax.transAxes, ha="center", va="center")
    _save(fig, path)


def plot_cost_vs_stability(tradeoff: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (ev_size, group) in enumerate(tradeoff.groupby("evaluation_size")):
        ax.plot(group["mean_seconds_per_run"], group["jaccard_top5_mean"], marker="o",
                color=COLOURS[i], label=f"evaluation size {ev_size}")
        for _, row in group.iterrows():
            ax.annotate(f"bg {int(row['background_size'])}",
                        (row["mean_seconds_per_run"], row["jaccard_top5_mean"]),
                        textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_xlabel("Mean SHAP seconds per run")
    ax.set_ylabel("Top-5 Jaccard (mean)")
    ax.set_title("Cost vs top-5 stability")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_analysis(grid_dir: Path, output_dir: Path, n_boot: int = 2000) -> dict:
    data = load_grid(grid_dir)
    within, between, consensus = data["within"], data["between"], data["consensus"]
    long, runs, reference = data["long"], data["runs"], data["reference"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    non_reference = between[~between["is_reference"].astype(bool)]
    effects_within = factor_effects(within, WITHIN_METRICS)
    effects_between = factor_effects(non_reference, BETWEEN_METRICS)
    kw, dunn = kruskal_and_dunn(within)
    tradeoff = size_tradeoff(within, runs)
    paired = size_paired_tests(within)
    regression = size_regression(within)
    boundary = top5_boundary(consensus, within)
    importance = feature_importance_by_method(consensus, within)
    topk = topk_changes_vs_reference(consensus, within, reference)
    swaps_by_run, swap_summary = top5_swaps(long, consensus, reference)
    ci = bootstrap_ci(long, n_boot=n_boot)

    outputs = {
        "factor_effects_within.csv": effects_within,
        "factor_effects_between.csv": effects_between,
        "kruskal_wallis.csv": kw,
        "dunn_posthoc.csv": dunn,
        "size_tradeoff.csv": tradeoff,
        "size_paired_tests.csv": paired,
        "size_regression.csv": regression,
        "top5_boundary.csv": boundary,
        "feature_importance_by_method.csv": importance,
        "topk_changes_vs_reference.csv": topk,
        "top5_swaps_by_run.csv": swaps_by_run,
        "top5_swap_summary.csv": swap_summary,
        "bootstrap_ci.csv": ci,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    plot_stability_vs_size(within, output_dir / "stability_vs_background_size.png")
    plot_top5_by_method(within, output_dir / "top5_by_evaluation_method.png")
    plot_temperature_importance(consensus, within, output_dir / "temperature_importance_by_method.png")
    plot_cost_vs_stability(tradeoff, output_dir / "cost_vs_stability.png")

    reference_top5 = list(consensus[reference].nlargest(5).index)
    summary = {
        "grid_dir": str(grid_dir),
        "configurations": int(len(within)),
        "runs": int(len(runs)),
        "reference_configuration": reference,
        "reference_top5": reference_top5,
        "runs_with_top5_differing_from_reference": int(swaps_by_run["differs"].sum()),
        "near_tie_configs_5_6": int(boundary["near_tie_5_6"].sum()),
        "near_tie_threshold_mw": NEAR_TIE_MW,
        "consensus_top5_matches_reference_everywhere": bool(
            (non_reference["jaccard_top5_vs_reference"] == 1.0).all()),
        "kendalls_w_by_background_size": within.groupby("background_size")["kendalls_w"].mean().round(4).to_dict(),
        "top5_jaccard_by_evaluation_method": within.groupby("evaluation_method")["jaccard_top5_mean"].mean().round(4).to_dict(),
        "bootstrap_resamples": n_boot,
    }
    summary = json.loads(json.dumps(summary, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=Path, default=Path("artifacts/shap_week10_grid"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis_week10"))
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    summary = run_analysis(args.grid, args.output, args.n_boot)
    print(json.dumps(summary, indent=2))
    print(f"\nTables and figures written to {args.output}")


if __name__ == "__main__":
    main()
