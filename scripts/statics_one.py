import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Input CSV for one sample. Expected columns:
# experiment_id, seed, alpha, chamfer_dist, emd
csv_path = "result/eye_04/experiment_errors.csv"  # TODO: update the path
sample_name = "sample_04"  # Used in output filenames and plot titles


# Load one sample's experiment results.
df = pd.read_csv(csv_path)
print("rows:", len(df))
print("columns:", df.columns.tolist())
print("unique experiment_id (20 runs expected):", df["experiment_id"].nunique())
print("alphas:", sorted(df["alpha"].unique()))

# Aggregate trials by alpha for this sample.
alpha_stats = (
    df.groupby("alpha")
      .agg(
          cd_mean=("chamfer_dist", "mean"),
          cd_std=("chamfer_dist", "std"),
          emd_mean=("emd", "mean"),
          emd_std=("emd", "std"),
      )
      .reset_index()
      .sort_values("alpha")
)

print("\nPer-alpha stats for this sample:")
print(alpha_stats)


# Labels for each alpha setting.
alpha_to_setting = {
    -1.0: "Coord-only",
     0.0: "Coord+RandomNoise",
     0.1: "Coord+Metric (α=0.1)",
     0.5: "Coord+Metric (α=0.5)",
     1.0: "Coord+Metric (α=1)",
     2.0: "Coord+Metric (α=2)",
    10.0: "Coord+Metric (α=10)",
    # Add alpha_star here if used, e.g.:
    # 3.557e-1: "Coord+Metric (α*)",
}

alpha_stats["setting"] = alpha_stats["alpha"].map(alpha_to_setting).fillna("Metric")


# Main ablation table for selected alpha values.
selected_alphas = [-1.0, 0.0, 0.1, 0.5, 1.0, 2, 10.0]  # Edit as needed

main_table = (
    alpha_stats[alpha_stats["alpha"].isin(selected_alphas)]
    .sort_values("alpha")
    .copy()
)

def fmt_mean_std(mean, std, scale=1.0):
    """Format mean+-std; scale can rescale values for table units."""
    mean_s = mean * scale
    std_s = std * scale
    return f"{mean_s:.3e} ± {std_s:.1e}"

main_table["CD (mean±std)"] = [
    fmt_mean_std(m, s, scale=1.0) for m, s in zip(main_table["cd_mean"], main_table["cd_std"])
]
main_table["EMD (mean±std)"] = [
    fmt_mean_std(m, s, scale=1.0) for m, s in zip(main_table["emd_mean"], main_table["emd_std"])
]

print("\n=== Main ablation table for this sample ===")
print(main_table[["setting", "alpha", "CD (mean±std)", "EMD (mean±std)"]])


# Generate a LaTeX table body for tabular.
cols_for_latex = main_table[["setting", "alpha", "cd_mean", "cd_std", "emd_mean", "emd_std"]].copy()

cols_for_latex["cd_str"] = cols_for_latex.apply(
    lambda r: fmt_mean_std(r["cd_mean"], r["cd_std"], scale=1.0), axis=1
)
cols_for_latex["emd_str"] = cols_for_latex.apply(
    lambda r: fmt_mean_std(r["emd_mean"], r["emd_std"], scale=1.0), axis=1
)

print("\n% ===== LaTeX table body for this sample =====\n")
for _, r in cols_for_latex.iterrows():
    line = f"{r['setting']} & {r['alpha']:.3g} & {r['cd_str']} & {r['emd_str']} \\\\"
    print(line)


# Plot alpha sensitivity curves for CD and EMD.
plt.figure()
plt.errorbar(
    alpha_stats["alpha"],
    alpha_stats["cd_mean"],
    yerr=alpha_stats["cd_std"],
    marker="o",
    linestyle="-",
)
plt.xscale("log")
plt.xlabel(r"$\alpha$")
plt.ylabel("Chamfer distance (mean ± std)")
plt.title(f"Sensitivity of Chamfer to $\\alpha$ ({sample_name})")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{sample_name}_alpha_sensitivity_cd.png", dpi=300)

plt.figure()
plt.errorbar(
    alpha_stats["alpha"],
    alpha_stats["emd_mean"],
    yerr=alpha_stats["emd_std"],
    marker="o",
    linestyle="-",
)
plt.xscale("log")
plt.xlabel(r"$\alpha$")
plt.ylabel("EMD (mean ± std)")
plt.title(f"Sensitivity of EMD to $\\alpha$ ({sample_name})")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{sample_name}_alpha_sensitivity_emd.png", dpi=300)
