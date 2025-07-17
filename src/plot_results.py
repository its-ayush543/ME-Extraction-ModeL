# plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load experiment results
df = pd.read_csv("results/surrogate_experiment_metrics.csv")

# Sanity check
print(df.head())


# Accuracy vs Fidelity (line plot)
plt.figure(figsize=(6, 5))
sns.lineplot(
    data=df,
    x="fidelity",
    y="accuracy",
    hue="query_size",       # Change to "noise" or "arch" if preferred
    style="arch",           # Change to "noise" or remove if not needed
    markers=True,
    dashes=False
)
plt.title("Accuracy vs Fidelity (Line Plot)")
plt.xlabel("Fidelity with Target Model")
plt.ylabel("Accuracy on True Labels")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/acc_vs_fid_line.png")
plt.show()

# 2. Accuracy vs Query Size
plt.figure(figsize=(6, 5))
sns.lineplot(data=df, x="query_size", y="accuracy", hue="arch", style="noise", markers=True, dashes=False)
plt.title("Accuracy vs Query Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/accuracy_vs_querysize.png")
plt.show()

# 3. Fidelity vs Query Size
plt.figure(figsize=(6, 5))
sns.lineplot(data=df, x="query_size", y="fidelity", hue="arch", style="noise", markers=True, dashes=False)
plt.title("Fidelity vs Query Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/fidelity_vs_querysize.png")
plt.show()

# 4. Bar Plot: Accuracy for each architecture at each noise level
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="arch", y="accuracy", hue="noise")
plt.title("Accuracy by Architecture and Noise")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("results/acc_by_arch_noise.png")
plt.show()
