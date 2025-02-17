import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import pandas as pd


path = Path("igs2gs/igs2gs_metrics/charts/metrics_all.csv")
df = pd.read_csv(path)
# df = df[~df["SA"].isin(["1", "17", "33", "49", "5", "21", "37", "53", "8", "24", "40", "56"])]

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(40, 10))
df.sort_values(by=["Quality", "SA"], inplace=True)
y_lim_max = 0.07
y_lim_min = 0.01

box = sns.boxplot(data=df, x="Quality", y="Mean LPIPS", hue="Quality", palette="mako", ax=axes[0])
box.set_ylim(y_lim_min, y_lim_max)
bar = sns.barplot(data=df, x="SA", y="Mean LPIPS", hue="Quality", palette="mako", ax=axes[1])
bar.set_ylim(y_lim_min, y_lim_max)

sns.move_legend(
    box,
    "upper center",
    ncol=5,
    title=None,
    frameon=False,
)
sns.move_legend(
    bar,
    "upper center",
    ncol=5,
    title=None,
    frameon=False,
)
# plt.rcParams["figure.figsize"] = (800, 400)

# Save the plot
plt.tight_layout()

# plt.show()

plt.savefig("igs2gs/igs2gs_metrics/charts/LPIPS_all.png")
plt.clf()
