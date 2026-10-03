"""Charts and a markdown summary written to the reports folder."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

INK, BLUE, ORANGE, GREY = "#1B1F2A", "#0050E6", "#FF5A00", "#9AA3B2"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def roc_chart(y, scores: dict, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for (name, s), color in zip(scores.items(), [GREY, BLUE]):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=color, lw=2, label=name)
    ax.plot([0, 1], [0, 1], ls=":", color=GREY)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False)
    _style(ax, "ROC on held-out test set")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def tier_chart(summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(summary["tier"], summary["observed_default_rate"] * 100, color=[BLUE] * 4 + [ORANGE])
    for x, v in zip(summary["tier"], summary["observed_default_rate"] * 100):
        ax.text(x, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Observed default rate (%)"); ax.set_xlabel("Risk tier")
    _style(ax, "Default rate rises monotonically by tier")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def tradeoff_chart(sim: pd.DataFrame, chosen: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sim["approval_rate"] * 100, sim["expected_loss_rate"] * 100, color=BLUE, lw=2, label="Expected loss")
    ax.plot(sim["approval_rate"] * 100, sim["realised_loss_rate"] * 100, color=GREY, lw=1.5, ls="--", label="Realised loss")
    ax.scatter([chosen["approval_rate"] * 100], [chosen["expected_loss_rate"] * 100], color=ORANGE, zorder=3, s=60,
               label="Recommended cut-off")
    ax.set_xlabel("Approval rate (%)"); ax.set_ylabel("Loss rate on approved book (%)")
    ax.legend(frameon=False)
    _style(ax, "Approval rate vs expected loss")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def importance_chart(importance: pd.Series, path: Path, top: int = 15):
    imp = importance.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("Total gain")
    _style(ax, f"Top {top} LightGBM features")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def write_summary(out: Path, meta: dict, metrics: dict, tiers: pd.DataFrame, chosen: pd.Series):
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "metrics": metrics}, indent=2))
    t = tiers.copy()
    for c in ("pd_ceiling", "avg_pd", "observed_default_rate", "share", "avg_apr"):
        t[c] = (t[c] * 100).round(1).astype(str) + "%"
    lines = [
        "# Results",
        "",
        f"Data source: **{meta['source']}**, {meta['applicants']:,} applicants, "
        f"default rate {meta['default_rate']:.1%}, thin file share {meta['thin_file_share']:.1%}.",
        "",
        "## Model performance (held-out test set)",
        "",
        "| Model | AUC | Gini | KS |",
        "|---|---|---|---|",
        *[f"| {k} | {v['AUC']:.3f} | {v['Gini']:.3f} | {v['KS']:.3f} |" for k, v in metrics.items()],
        "",
        "## Risk tiers",
        "",
        t.to_markdown(index=False),
        "",
        "## Recommended cut-off",
        "",
        f"Approve applicants with PD at or below **{chosen['pd_cutoff']:.1%}**: approval rate "
        f"**{chosen['approval_rate']:.1%}**, expected loss rate **{chosen['expected_loss_rate']:.2%}**, "
        f"realised loss rate {chosen['realised_loss_rate']:.2%}, bad rate {chosen['bad_rate_approved']:.1%}.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines))
