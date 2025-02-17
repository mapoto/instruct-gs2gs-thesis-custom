import matplotlib.pyplot as plt
import numpy as np


metric_type = ["ciede2000", "rmse", "lpips"]
human_models = ["ephra", "dora", "irene", "simon"]
style = ["stone", "red", "vangogh", "3d"]

m = metric_type[1]
h = human_models[0]


data = []
# Unstylized
data1 = np.genfromtxt(
    f"/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/high/{h}_elf/30000_{m}.csv",
    delimiter=",",
)
data.append(data1)
labels = ["Unstylized"]

for s in style:
    # Low Sp
    data2 = np.genfromtxt(
        f"/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/low/{h}_{s}/35000_{m}.csv",
        delimiter=",",
    )
    data.append(data2)
    labels.append(f"Low Sp {s}")

    # High Sp
    data3 = np.genfromtxt(
        f"/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/high/{h}_{s}/35000_{m}.csv",
        delimiter=",",
    )
    data.append(data3)
    labels.append(f"High Sp {s}")


# Combine data1 and data2 into a list

# Create the box plot
plt.boxplot(data, labels=labels)

# Add title and labels
plt.title("Human Model: A")
plt.ylabel("Values")

# Display the plot
plt.show()
