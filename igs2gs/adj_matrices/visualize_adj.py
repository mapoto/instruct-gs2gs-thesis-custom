import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

camera_names = [
    ("1-2-1", "1"),
    ("1-3-1", "2"),
    ("1-4-1", "3"),
    ("1-5-1", "4"),
    ("1-5-2", "5"),
    ("1-6-1", "6"),
    ("1-6-2", "7"),
    ("2-2-1", "8"),
    ("2-3-1", "9"),
    ("2-4-1", "10"),
    ("2-4-2", "11"),
    ("2-5-1", "12"),
    ("2-5-2", "13"),
    ("2-6-1", "14"),
    ("2-6-2", "15"),
    ("3-2-1", "16"),
    ("3-3-1", "17"),
    ("3-3-2", "18"),
    ("3-4-1", "19"),
    ("3-5-1", "20"),
    ("3-5-2", "21"),
    ("3-6-1", "22"),
    ("B-2-1", "23"),
    ("B-3-1", "24"),
    ("B-4-1", "25"),
    ("B-4-2", "26"),
    ("B-5-1", "27"),
    ("B-5-2", "28"),
    ("B-6-1", "29"),
    ("B-6-2", "30"),
    ("C-2-1", "31"),
    ("C-3-1", "32"),
    ("C-3-2", "33"),
    ("C-4-1", "34"),
    ("C-4-2", "35"),
    ("C-5-1", "36"),
    ("C-6-1", "37"),
]

# Load the CSV file into a DataFrame
df = pd.read_csv("igs2gs/adj_matrices/simon.csv", header=0)
# Convert the DataFrame to float type
df = df.astype(float)

# Set the diagonal values to NaN
np.fill_diagonal(df.values, np.nan)

# Create a custom color map
cmap = sns.color_palette("Blues", as_cmap=True)
cmap.set_bad(color="yellow")  # Set the color for NaN values

# Create a custom layout using GridSpec
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(1, 2, width_ratios=[15, 1], wspace=0.3)

# Create the heat map with the custom color map
ax0 = fig.add_subplot(gs[0])
sns.heatmap(
    df,
    cmap=cmap,
    annot=False,
    mask=df.isna(),
    cbar_kws={"label": "Adjacency"},
    yticklabels=np.arange(1, len(df) + 1),
    ax=ax0,
)

# Overlay the diagonal with yellow color
for i in range(len(df)):
    ax0.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="yellow", edgecolor="yellow"))

# Create a list of labels for the legend
labels = [f"{name[1]}: {name[0]}" for name in camera_names]

# Create a custom legend
legend_elements = [
    plt.Line2D([0], [0], color="w", label=label, markersize=10, markerfacecolor="gray") for label in labels
]

# Add the legend to the right side of the colorbar
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.legend(handles=legend_elements, title="Camera Names", loc="center", ncol=2)

# Display the heat map
plt.show()
