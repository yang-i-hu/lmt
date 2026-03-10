"""
Generate comparison charts for all models and save as PNG files.
Run from v2/: python docs/plot_results.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Results Data ──────────────────────────────────────────────────────────

models = ["TreeModel(基线)", "ElasticNet", "AutoEncoder", "DNN", "CrossSectional", "ResidualMLP", "Temporal"]

# Aggregate (all years) metrics  — TreeModel baseline first
ic_all    = [0.12060, 0.15387, 0.15416, 0.15195, 0.15274, 0.14814, 0.13452]
icir_all  = [1.45530, 1.83048, 1.89541, 1.84679, 1.85078, 1.89088, 1.71870]
ls_all    = [1.17590, 1.54848, 1.47565, 1.44417, 1.49913, 1.38536, 1.18016]
ir_ls_all = [9.63830, 14.13546, 14.50230, 14.33494, 14.37564, 12.97842, 11.51000]

# Per-year IC  (TreeModel has no per-year breakdown; use aggregate as placeholder)
ic_2019 = [None, 0.16945, 0.16750, 0.16280, 0.16619, 0.15526, 0.14019]
ic_2020 = [None, 0.15960, 0.15489, 0.15728, 0.15684, 0.15397, 0.13801]
ic_2021 = [None, 0.13248, 0.14031, 0.13573, 0.13518, 0.13515, 0.12533]

# Per-year Long-Short return  (TreeModel has no per-year breakdown)
ls_2019 = [None, 1.37562, 1.34606, 1.24629, 1.28950, 1.14997, 0.93894]
ls_2020 = [None, 1.67050, 1.46178, 1.47514, 1.58758, 1.43505, 1.22246]
ls_2021 = [None, 1.60938, 1.62547, 1.62648, 1.63565, 1.59268, 1.40475]

# Sort by IC (descending)
sort_idx = np.argsort(ic_all)[::-1]
models_sorted = [models[i] for i in sort_idx]
ic_sorted = [ic_all[i] for i in sort_idx]
icir_sorted = [icir_all[i] for i in sort_idx]
ls_sorted = [ls_all[i] for i in sort_idx]
ir_ls_sorted = [ir_ls_all[i] for i in sort_idx]

colors = ["#795548", "#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336", "#607D8B"]
colors_sorted = [colors[i] for i in sort_idx]

# Try to use a font that supports Chinese characters
import os as _os
for _font in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
              "Arial Unicode MS", "DejaVu Sans"]:
    try:
        from matplotlib.font_manager import FontProperties
        _fp = FontProperties(family=_font)
        if _fp.get_name() != _font and _font not in ("DejaVu Sans",):
            continue
        plt.rcParams["font.sans-serif"] = [_font] + plt.rcParams.get("font.sans-serif", [])
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.facecolor": "white",
})

# ═══ Chart 1: Aggregate Metrics (4 subplots) ═══════════════════════════

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Model Comparison \u2014 Aggregate Metrics (2019\u20132021)",
             fontsize=15, fontweight="bold", y=1.02)

metrics_data = [
    ("IC", ic_sorted), ("ICIR", icir_sorted),
    ("Long-Short Return", ls_sorted), ("Long-Short IR", ir_ls_sorted),
]
for ax, (title, vals) in zip(axes, metrics_data):
    bars = ax.barh(models_sorted[::-1], vals[::-1],
                   color=colors_sorted[::-1], edgecolor="white", height=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(min(vals) * 0.9, max(vals) * 1.05)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + (max(vals) - min(vals)) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}" if v < 2 else f"{v:.2f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("docs/model_comparison_aggregate.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: docs/model_comparison_aggregate.png")

# ═══ Chart 2: Per-Year IC ═══════════════════════════════════════════════
# Skip TreeModel (index 0) — no per-year breakdown available
models_nn = models[1:]
ic19_nn = [v for v in ic_2019 if v is not None]
ic20_nn = [v for v in ic_2020 if v is not None]
ic21_nn = [v for v in ic_2021 if v is not None]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models_nn))
width = 0.25
bars1 = ax.bar(x - width, ic19_nn, width, label="2019", color="#42A5F5", edgecolor="white")
bars2 = ax.bar(x,         ic20_nn, width, label="2020", color="#66BB6A", edgecolor="white")
bars3 = ax.bar(x + width, ic21_nn, width, label="2021", color="#FFA726", edgecolor="white")
# Add TreeModel baseline as dashed line
ax.axhline(y=ic_all[0], color="#795548", linestyle="--", linewidth=2,
           label=f"TreeModel baseline (IC={ic_all[0]:.4f})")
ax.set_ylabel("IC (Spearman)")
ax.set_title("Per-Year IC Comparison by Model", fontweight="bold", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models_nn, rotation=15)
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.10, 0.19)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("docs/model_ic_per_year.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: docs/model_ic_per_year.png")

# ═══ Chart 3: Per-Year LS Return ════════════════════════════════════════
ls19_nn = [v for v in ls_2019 if v is not None]
ls20_nn = [v for v in ls_2020 if v is not None]
ls21_nn = [v for v in ls_2021 if v is not None]

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, ls19_nn, width, label="2019", color="#42A5F5", edgecolor="white")
bars2 = ax.bar(x,         ls20_nn, width, label="2020", color="#66BB6A", edgecolor="white")
bars3 = ax.bar(x + width, ls21_nn, width, label="2021", color="#FFA726", edgecolor="white")
# Add TreeModel baseline as dashed line
ax.axhline(y=ls_all[0], color="#795548", linestyle="--", linewidth=2,
           label=f"TreeModel baseline (LS={ls_all[0]:.4f})")
ax.set_ylabel("Long-Short Return (%)")
ax.set_title("Per-Year Long-Short Return by Model", fontweight="bold", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models_nn, rotation=15)
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.7, 1.85)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("docs/model_ls_per_year.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: docs/model_ls_per_year.png")

# ═══ Chart 4: Radar ═════════════════════════════════════════════════════

categories = ["IC", "ICIR", "LS Return", "LS IR", "HS Short", "HS Long"]
hs_short_all = [0.06040, 0.06042, 0.06102, 0.06172, 0.06078, 0.06066, 0.05514]
hs_long_all  = [0.05770, 0.05725, 0.05822, 0.05782, 0.06011, 0.06293, 0.05378]

raw = np.array([ic_all, icir_all, ls_all, ir_ls_all, hs_short_all, hs_long_all]).T
mins, maxs = raw.min(axis=0), raw.max(axis=0)
normed = (raw - mins) / (maxs - mins + 1e-9)

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

for i, model in enumerate(models):
    values = normed[i].tolist() + [normed[i][0]]
    ax.plot(angles, values, "o-", linewidth=2, label=model, color=colors[i])
    ax.fill(angles, values, alpha=0.08, color=colors[i])

ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], categories)
ax.set_ylim(0, 1.1)
ax.set_title("Multi-Metric Radar Comparison (Normalised)",
             fontweight="bold", fontsize=13, pad=20)
ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05), fontsize=9)
plt.tight_layout()
plt.savefig("docs/model_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: docs/model_radar.png")

print("\nAll charts saved to docs/")
